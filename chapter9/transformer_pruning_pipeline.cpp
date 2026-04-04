#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// ---------------------------------------------------------------------------
// Transformer block: self-attention + FFN with residual connections
// ---------------------------------------------------------------------------
struct TransformerBlock : torch::nn::Module {
    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    int d_model, nhead, d_ff;

    TransformerBlock(int d_model_, int nhead_, int d_ff_)
        : d_model(d_model_), nhead(nhead_), d_ff(d_ff_) {
        attn = register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model_, nhead_)));
        norm1 = register_module("norm1", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d_model_})));
        norm2 = register_module("norm2", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d_model_})));
        fc1 = register_module("fc1", torch::nn::Linear(d_model_, d_ff_));
        fc2 = register_module("fc2", torch::nn::Linear(d_ff_, d_model_));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto attn_out = std::get<0>(attn->forward(x, x, x));
        x = norm1->forward(x + attn_out);
        auto ff_out = fc2->forward(torch::relu(fc1->forward(x)));
        return norm2->forward(x + ff_out);
    }
};

// ---------------------------------------------------------------------------
// Multi-layer transformer classifier
// ---------------------------------------------------------------------------
struct TransformerClassifier : torch::nn::Module {
    torch::nn::ModuleList layers{nullptr};
    torch::nn::Linear classifier{nullptr};
    int d_model, nhead, d_ff, num_layers;

    TransformerClassifier(int num_layers_, int d_model_, int nhead_, int d_ff_, int num_classes)
        : d_model(d_model_), nhead(nhead_), d_ff(d_ff_), num_layers(num_layers_) {
        layers = register_module("layers", torch::nn::ModuleList());
        for (int i = 0; i < num_layers_; ++i)
            layers->push_back(std::make_shared<TransformerBlock>(d_model_, nhead_, d_ff_));
        classifier = register_module("classifier",
            torch::nn::Linear(d_model_, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [batch, d_model] -> [1, batch, d_model]
        x = x.unsqueeze(0);
        for (size_t i = 0; i < layers->size(); ++i)
            x = layers->ptr<TransformerBlock>(i)->forward(x);
        return classifier->forward(x.squeeze(0));
    }

    // Forward skipping a specific layer index
    torch::Tensor forward_skip_layer(torch::Tensor x, int skip_idx) {
        x = x.unsqueeze(0);
        for (size_t i = 0; i < layers->size(); ++i) {
            if (static_cast<int>(i) == skip_idx) continue;
            x = layers->ptr<TransformerBlock>(i)->forward(x);
        }
        return classifier->forward(x.squeeze(0));
    }
};

// ---------------------------------------------------------------------------
// Utility: measure sparsity and accuracy
// ---------------------------------------------------------------------------
float measure_sparsity(torch::nn::Module& model) {
    int64_t total = 0, zeros = 0;
    for (const auto& p : model.parameters()) {
        total += p.numel();
        zeros += (p.data() == 0).sum().item<int64_t>();
    }
    return static_cast<float>(zeros) / total;
}

float evaluate(TransformerClassifier& model, torch::Tensor data, torch::Tensor labels) {
    torch::NoGradGuard no_grad;
    auto preds = model.forward(data).argmax(1);
    return preds.eq(labels).sum().template item<float>() / labels.size(0) * 100;
}

// ---------------------------------------------------------------------------
// 1. Attention Head Pruning
//    Importance = L2 norm of each head's Q/K/V projection slice
// ---------------------------------------------------------------------------
class AttentionHeadPruner {
public:
    static std::vector<float> compute_head_importance(TransformerBlock& block) {
        torch::NoGradGuard no_grad;
        int num_heads = block.nhead;
        int d_k = block.d_model / num_heads;
        auto in_proj = block.attn->in_proj_weight;
        int dm = block.d_model;

        std::vector<float> importance(num_heads);
        for (int h = 0; h < num_heads; ++h) {
            int s = h * d_k, e = s + d_k;
            float q = in_proj.slice(0, s, e).norm().template item<float>();
            float k = in_proj.slice(0, dm + s, dm + e).norm().template item<float>();
            float v = in_proj.slice(0, 2 * dm + s, 2 * dm + e).norm().template item<float>();
            importance[h] = q + k + v;
        }
        return importance;
    }

    static std::vector<int> prune_heads(TransformerBlock& block, int num_to_prune) {
        auto importance = compute_head_importance(block);
        int num_heads = block.nhead;
        int d_k = block.d_model / num_heads;
        int dm = block.d_model;

        std::vector<int> indices(num_heads);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return importance[a] < importance[b]; });

        std::vector<int> pruned(indices.begin(), indices.begin() + num_to_prune);

        torch::NoGradGuard no_grad;
        auto& w = block.attn->in_proj_weight;
        auto& b = block.attn->in_proj_bias;
        auto& ow = block.attn->out_proj->weight;

        for (int h : pruned) {
            int s = h * d_k, e = s + d_k;
            w.slice(0, s, e).zero_();
            w.slice(0, dm + s, dm + e).zero_();
            w.slice(0, 2 * dm + s, 2 * dm + e).zero_();
            if (b.defined()) {
                b.slice(0, s, e).zero_();
                b.slice(0, dm + s, dm + e).zero_();
                b.slice(0, 2 * dm + s, 2 * dm + e).zero_();
            }
            ow.slice(1, s, e).zero_();
        }
        return pruned;
    }
};

// ---------------------------------------------------------------------------
// 2. FFN Neuron Pruning
//    Importance = L1 norm of each intermediate neuron's weights in fc1
//    Prunes by zeroing rows in fc1 and corresponding columns in fc2
// ---------------------------------------------------------------------------
class FFNPruner {
public:
    static torch::Tensor compute_neuron_importance(TransformerBlock& block) {
        torch::NoGradGuard no_grad;
        // fc1.weight: [d_ff, d_model] — each row is one neuron
        return block.fc1->weight.abs().sum(1); // [d_ff]
    }

    static int prune_neurons(TransformerBlock& block, float prune_ratio) {
        torch::NoGradGuard no_grad;
        auto importance = compute_neuron_importance(block);
        int64_t d_ff = importance.size(0);
        int64_t prune_count = static_cast<int64_t>(d_ff * prune_ratio);

        // Find threshold via k-th smallest
        auto [sorted_vals, sorted_idx] = importance.sort();
        float threshold = sorted_vals[prune_count].template item<float>();

        auto prune_mask = importance < threshold; // neurons to zero

        // Zero pruned rows in fc1 weight/bias and corresponding columns in fc2
        for (int64_t n = 0; n < d_ff; ++n) {
            if (prune_mask[n].template item<bool>()) {
                block.fc1->weight[n].zero_();
                if (block.fc1->bias.defined())
                    block.fc1->bias[n].zero_();
                block.fc2->weight.select(1, n).zero_();
            }
        }
        return static_cast<int>(prune_count);
    }
};

// ---------------------------------------------------------------------------
// 3. Layer Pruning
//    Importance = mean absolute output change when layer is skipped
// ---------------------------------------------------------------------------
class LayerPruner {
public:
    static std::vector<float> compute_layer_importance(
        TransformerClassifier& model, torch::Tensor data
    ) {
        torch::NoGradGuard no_grad;
        int n = static_cast<int>(model.layers->size());
        auto full_output = model.forward(data);
        std::vector<float> importance(n);

        for (int i = 0; i < n; ++i) {
            auto skip_output = model.forward_skip_layer(data, i);
            importance[i] = (full_output - skip_output).abs().mean().template item<float>();
        }
        return importance;
    }

    // Zero out all parameters of the least important layer (effectively skipping it)
    static int prune_least_important_layer(
        TransformerClassifier& model, torch::Tensor data,
        std::vector<bool>& pruned_layers
    ) {
        auto importance = compute_layer_importance(model, data);
        // Find least important among non-pruned layers
        int min_idx = -1;
        float min_val = std::numeric_limits<float>::max();
        for (size_t i = 0; i < importance.size(); ++i) {
            if (!pruned_layers[i] && importance[i] < min_val) {
                min_val = importance[i];
                min_idx = static_cast<int>(i);
            }
        }

        // Zero attention + FFN weights so the block becomes identity via residuals
        // Keep LayerNorm intact so residual x passes through correctly
        torch::NoGradGuard no_grad;
        auto& block = *model.layers->ptr<TransformerBlock>(min_idx);
        block.attn->in_proj_weight.zero_();
        if (block.attn->in_proj_bias.defined())
            block.attn->in_proj_bias.zero_();
        block.attn->out_proj->weight.zero_();
        if (block.attn->out_proj->bias.defined())
            block.attn->out_proj->bias.zero_();
        block.fc1->weight.zero_();
        if (block.fc1->bias.defined())
            block.fc1->bias.zero_();
        block.fc2->weight.zero_();
        if (block.fc2->bias.defined())
            block.fc2->bias.zero_();
        pruned_layers[min_idx] = true;
        return min_idx;
    }
};

// ---------------------------------------------------------------------------
// Main: train, then demonstrate all three pruning strategies
// ---------------------------------------------------------------------------
int main() {
    torch::manual_seed(42);

    int num_layers = 4, d_model = 64, nhead = 8, d_ff = 128;
    int num_classes = 5, num_samples = 200;
    auto data = torch::randn({num_samples, d_model});
    auto labels = torch::randint(0, num_classes, {num_samples});

    auto model = std::make_shared<TransformerClassifier>(
        num_layers, d_model, nhead, d_ff, num_classes);

    // Train
    torch::optim::Adam optimizer(model->parameters(), 0.001);
    std::cout << "Training transformer model (" << num_layers << " layers)..." << std::endl;
    for (int epoch = 0; epoch < 50; ++epoch) {
        optimizer.zero_grad();
        auto loss = torch::nn::functional::cross_entropy(model->forward(data), labels);
        loss.backward();
        optimizer.step();
        if (epoch % 10 == 0)
            std::cout << "  Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }

    float baseline_acc = evaluate(*model, data, labels);
    int64_t total_params = 0;
    for (const auto& p : model->parameters()) total_params += p.numel();
    std::cout << "\nBaseline - Accuracy: " << baseline_acc
              << "% | Parameters: " << total_params << std::endl;

    // Save weights for restoration between experiments
    std::vector<torch::Tensor> original_weights;
    for (const auto& p : model->parameters())
        original_weights.push_back(p.data().clone());

    auto restore_weights = [&]() {
        size_t idx = 0;
        for (auto& p : model->parameters())
            p.data().copy_(original_weights[idx++]);
    };

    // -----------------------------------------------------------------------
    // 1. Attention Head Pruning
    // -----------------------------------------------------------------------
    std::cout << "\n=== Attention Head Pruning ===" << std::endl;
    for (int heads_to_prune : {2, 4, 6}) {
        restore_weights();
        std::cout << "\nPruning " << heads_to_prune << "/" << nhead << " heads per layer:";
        for (size_t l = 0; l < model->layers->size(); ++l) {
            auto& block = *model->layers->ptr<TransformerBlock>(l);
            auto pruned = AttentionHeadPruner::prune_heads(block, heads_to_prune);
            std::cout << " L" << l << "(";
            for (size_t i = 0; i < pruned.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << pruned[i];
            }
            std::cout << ")";
        }
        float sp = measure_sparsity(*model);
        float acc = evaluate(*model, data, labels);
        std::cout << "\n  Sparsity: " << sp * 100 << "% | Accuracy: " << acc << "%" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 2. FFN Neuron Pruning
    // -----------------------------------------------------------------------
    std::cout << "\n=== FFN Neuron Pruning ===" << std::endl;
    for (float ratio : {0.25f, 0.50f, 0.75f}) {
        restore_weights();
        int total_pruned = 0;
        for (size_t l = 0; l < model->layers->size(); ++l) {
            auto& block = *model->layers->ptr<TransformerBlock>(l);
            total_pruned += FFNPruner::prune_neurons(block, ratio);
        }
        float sp = measure_sparsity(*model);
        float acc = evaluate(*model, data, labels);
        std::cout << "FFN prune ratio: " << ratio * 100 << "%"
                  << " | Neurons zeroed: " << total_pruned
                  << " | Sparsity: " << sp * 100 << "%"
                  << " | Accuracy: " << acc << "%" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 3. Layer Pruning
    // -----------------------------------------------------------------------
    std::cout << "\n=== Layer Pruning ===" << std::endl;
    // Need a fresh model since layer pruning modifies structure
    auto layer_model = std::make_shared<TransformerClassifier>(
        num_layers, d_model, nhead, d_ff, num_classes);

    // Copy trained weights into layer_model
    {
        auto src = model->parameters();
        auto dst = layer_model->parameters();
        for (size_t i = 0; i < src.size(); ++i)
            dst[i].data().copy_(original_weights[i]);
    }

    auto layer_importance = LayerPruner::compute_layer_importance(*layer_model, data);
    std::cout << "Layer importance scores:" << std::endl;
    for (size_t i = 0; i < layer_importance.size(); ++i)
        std::cout << "  Layer " << i << ": " << layer_importance[i] << std::endl;

    // Iteratively prune least important layers
    std::vector<bool> pruned_layers(num_layers, false);
    for (int r = 0; r < num_layers - 1; ++r) {
        int removed = LayerPruner::prune_least_important_layer(*layer_model, data, pruned_layers);
        int remaining = num_layers - (r + 1);
        float acc = evaluate(*layer_model, data, labels);
        std::cout << "Zeroed layer " << removed
                  << " | Active layers: " << remaining
                  << " | Accuracy: " << acc << "%" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 4. Combined Pipeline: head + FFN + layer pruning
    // -----------------------------------------------------------------------
    std::cout << "\n=== Combined Pruning Pipeline ===" << std::endl;
    auto combined = std::make_shared<TransformerClassifier>(
        num_layers, d_model, nhead, d_ff, num_classes);
    {
        auto dst = combined->parameters();
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i].data().copy_(original_weights[i]);
    }
    std::cout << "Baseline accuracy: " << evaluate(*combined, data, labels) << "%" << std::endl;

    // Step 1: Prune 1 layer
    std::vector<bool> combined_pruned(num_layers, false);
    int removed = LayerPruner::prune_least_important_layer(*combined, data, combined_pruned);
    std::cout << "After pruning layer " << removed << ": "
              << evaluate(*combined, data, labels) << "%" << std::endl;

    // Step 2: Prune 4/8 heads in remaining layers
    for (size_t l = 0; l < combined->layers->size(); ++l)
        AttentionHeadPruner::prune_heads(*combined->layers->ptr<TransformerBlock>(l), 4);
    std::cout << "After pruning 4/8 heads: "
              << evaluate(*combined, data, labels) << "%" << std::endl;

    // Step 3: Prune 50% FFN neurons in remaining layers
    for (size_t l = 0; l < combined->layers->size(); ++l)
        FFNPruner::prune_neurons(*combined->layers->ptr<TransformerBlock>(l), 0.5f);
    float final_sp = measure_sparsity(*combined);
    float final_acc = evaluate(*combined, data, labels);
    std::cout << "After pruning 50% FFN neurons: " << final_acc << "%"
              << " | Final sparsity: " << final_sp * 100 << "%" << std::endl;

    return 0;
}
