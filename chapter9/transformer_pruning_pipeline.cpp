// transformer_pruning_pipeline.cpp
//
// Comprehensive transformer pruning pipeline demonstrating three complementary
// structured pruning strategies applied to a multi-layer transformer classifier:
//
//   1. Attention Head Pruning  - removes redundant heads in multi-head attention
//   2. FFN Neuron Pruning      - removes unimportant neurons in feed-forward layers
//   3. Layer Pruning            - removes entire transformer blocks
//
// Each strategy exploits the modular structure of transformers to achieve
// structured sparsity (entire rows/columns/blocks zeroed), which is more
// hardware-friendly than unstructured (individual weight) pruning.
//
// The program trains a 4-layer transformer, runs each strategy independently,
// then demonstrates a combined pipeline applying all three sequentially.
//
// Dependencies: LibTorch (PyTorch C++ API)
// Compile:
//   g++ -std=c++17 transformer_pruning_pipeline.cpp -o transformer_pruning_pipeline \
//     -I$LIBTORCH_PATH/include -I$LIBTORCH_PATH/include/torch/csrc/api/include \
//     -L$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$LIBTORCH_PATH/lib

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// ---------------------------------------------------------------------------
// TransformerBlock: single encoder layer
//
// Architecture (standard pre-norm transformer):
//   x -> MultiHeadAttention(x, x, x) -> Add & LayerNorm -> FFN -> Add & LayerNorm
//
// The FFN follows: FFN(x) = W2 * ReLU(W1 * x + b1) + b2
//   - fc1: [d_model -> d_ff]   (expansion layer)
//   - fc2: [d_ff -> d_model]   (projection layer)
//
// The residual connections (x + sublayer(x)) are critical for layer pruning:
// zeroing the sublayer weights makes the block pass input through unchanged.
// ---------------------------------------------------------------------------
struct TransformerBlock : torch::nn::Module {
    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    int d_model, nhead, d_ff;

    TransformerBlock(int d_model_, int nhead_, int d_ff_)
        : d_model(d_model_), nhead(nhead_), d_ff(d_ff_) {
        // Multi-head self-attention with nhead parallel heads
        // Internally stores in_proj_weight [3*d_model, d_model] = stacked W_Q, W_K, W_V
        // and out_proj [d_model, d_model] for the output projection
        attn = register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model_, nhead_)));
        norm1 = register_module("norm1", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d_model_})));
        norm2 = register_module("norm2", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d_model_})));
        // FFN intermediate layer expands from d_model to d_ff
        fc1 = register_module("fc1", torch::nn::Linear(d_model_, d_ff_));
        // FFN output layer projects back from d_ff to d_model
        fc2 = register_module("fc2", torch::nn::Linear(d_ff_, d_model_));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Self-attention sublayer with residual connection
        auto attn_out = std::get<0>(attn->forward(x, x, x));
        x = norm1->forward(x + attn_out);  // Add & Norm

        // FFN sublayer with residual connection
        auto ff_out = fc2->forward(torch::relu(fc1->forward(x)));
        return norm2->forward(x + ff_out);  // Add & Norm
    }
};

// ---------------------------------------------------------------------------
// TransformerClassifier: stack of TransformerBlocks + linear classification head
//
// Input:  [batch, d_model] (each sample is a d_model-dimensional vector)
// Output: [batch, num_classes] (raw logits for classification)
//
// For attention, input is reshaped to [seq_len=1, batch, d_model] since
// LibTorch's MultiheadAttention expects [seq_len, batch, d_model].
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
        x = x.unsqueeze(0); // [batch, d_model] -> [1, batch, d_model]
        for (size_t i = 0; i < layers->size(); ++i)
            x = layers->ptr<TransformerBlock>(i)->forward(x);
        return classifier->forward(x.squeeze(0)); // [batch, num_classes]
    }

    // Forward that skips one layer — used by LayerPruner to measure
    // how much the output changes when a layer is removed
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
// Utility functions
// ---------------------------------------------------------------------------

// Fraction of model parameters that are exactly zero
float measure_sparsity(torch::nn::Module& model) {
    int64_t total = 0, zeros = 0;
    for (const auto& p : model.parameters()) {
        total += p.numel();
        zeros += (p.data() == 0).sum().item<int64_t>();
    }
    return static_cast<float>(zeros) / total;
}

// Classification accuracy (%) on given data/labels
float evaluate(TransformerClassifier& model, torch::Tensor data, torch::Tensor labels) {
    torch::NoGradGuard no_grad;
    auto preds = model.forward(data).argmax(1);
    return preds.eq(labels).sum().template item<float>() / labels.size(0) * 100;
}

// ---------------------------------------------------------------------------
// Strategy 1: Attention Head Pruning
//
// Multi-head attention computes h parallel attention operations. Research
// (Michel et al., 2019) shows many heads are redundant and can be removed
// with minimal accuracy loss.
//
// How it works:
//   - LibTorch stores Q, K, V projections in a single in_proj_weight tensor
//     of shape [3*d_model, d_model], laid out as [W_Q; W_K; W_V]
//   - Each head h owns rows [h*d_k : (h+1)*d_k] within each of the three
//     W_Q, W_K, W_V blocks, where d_k = d_model / nhead
//   - Importance = L2 norm of head's Q + K + V projection weights
//     (larger norm = head contributes more to the output)
//   - Pruning zeroes the head's rows in in_proj_weight, its biases,
//     and the corresponding columns in out_proj (output projection)
//
// Memory layout of in_proj_weight [3*d_model, d_model]:
//   rows [0, d_model)           -> W_Q (all heads)
//   rows [d_model, 2*d_model)   -> W_K (all heads)
//   rows [2*d_model, 3*d_model) -> W_V (all heads)
//
// Within each block, head h occupies rows [h*d_k, (h+1)*d_k).
// ---------------------------------------------------------------------------
class AttentionHeadPruner {
public:
    // Score each head by the combined L2 norm of its Q, K, V weight slices
    static std::vector<float> compute_head_importance(TransformerBlock& block) {
        torch::NoGradGuard no_grad;
        int num_heads = block.nhead;
        int d_k = block.d_model / num_heads; // per-head dimension
        auto in_proj = block.attn->in_proj_weight;
        int dm = block.d_model;

        std::vector<float> importance(num_heads);
        for (int h = 0; h < num_heads; ++h) {
            int s = h * d_k, e = s + d_k;
            // L2 norm of this head's Q projection rows
            float q = in_proj.slice(0, s, e).norm().template item<float>();
            // L2 norm of this head's K projection rows
            float k = in_proj.slice(0, dm + s, dm + e).norm().template item<float>();
            // L2 norm of this head's V projection rows
            float v = in_proj.slice(0, 2 * dm + s, 2 * dm + e).norm().template item<float>();
            importance[h] = q + k + v;
        }
        return importance;
    }

    // Zero out the num_to_prune least important heads' weights
    static std::vector<int> prune_heads(TransformerBlock& block, int num_to_prune) {
        auto importance = compute_head_importance(block);
        int num_heads = block.nhead;
        int d_k = block.d_model / num_heads;
        int dm = block.d_model;

        // Sort head indices by importance (ascending) — least important first
        std::vector<int> indices(num_heads);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return importance[a] < importance[b]; });

        // Select the least important heads for pruning
        std::vector<int> pruned(indices.begin(), indices.begin() + num_to_prune);

        torch::NoGradGuard no_grad;
        auto& w = block.attn->in_proj_weight;  // [3*d_model, d_model]
        auto& b = block.attn->in_proj_bias;    // [3*d_model]
        auto& ow = block.attn->out_proj->weight; // [d_model, d_model]

        for (int h : pruned) {
            int s = h * d_k, e = s + d_k;

            // Zero this head's Q, K, V rows in the combined projection
            w.slice(0, s, e).zero_();           // Q rows
            w.slice(0, dm + s, dm + e).zero_(); // K rows
            w.slice(0, 2 * dm + s, 2 * dm + e).zero_(); // V rows

            // Zero corresponding bias entries
            if (b.defined()) {
                b.slice(0, s, e).zero_();
                b.slice(0, dm + s, dm + e).zero_();
                b.slice(0, 2 * dm + s, 2 * dm + e).zero_();
            }

            // Zero output projection columns so this head's output is suppressed
            // out_proj multiplies concatenated head outputs: O = concat(h1,...,hn) * W_o
            // Zeroing columns [s:e] removes head h's contribution
            ow.slice(1, s, e).zero_();
        }
        return pruned;
    }
};

// ---------------------------------------------------------------------------
// Strategy 2: FFN Neuron Pruning
//
// Each transformer block has a two-layer FFN:
//   FFN(x) = W2 * ReLU(W1 * x + b1) + b2
//
// where fc1 (W1) has shape [d_ff, d_model] and fc2 (W2) has shape [d_model, d_ff].
// Each of the d_ff intermediate neurons corresponds to:
//   - One row in fc1.weight (input connections)
//   - One element in fc1.bias
//   - One column in fc2.weight (output connections)
//
// Importance metric: L1 norm of each neuron's input weights (fc1 row).
// Neurons with small L1 norm contribute little regardless of input.
//
// Pruning zeroes the neuron's fc1 row, fc1 bias, and fc2 column,
// effectively removing it from the computation.
// ---------------------------------------------------------------------------
class FFNPruner {
public:
    // L1 norm of each neuron's incoming weights in fc1
    static torch::Tensor compute_neuron_importance(TransformerBlock& block) {
        torch::NoGradGuard no_grad;
        // fc1.weight shape: [d_ff, d_model] — row n = neuron n's weights
        return block.fc1->weight.abs().sum(1); // [d_ff]
    }

    // Zero the least important neurons (by ratio) in the FFN
    // Returns the number of neurons pruned
    static int prune_neurons(TransformerBlock& block, float prune_ratio) {
        torch::NoGradGuard no_grad;
        auto importance = compute_neuron_importance(block);
        int64_t d_ff = importance.size(0);
        int64_t prune_count = static_cast<int64_t>(d_ff * prune_ratio);

        // Sort to find the threshold: neurons below this are pruned
        auto [sorted_vals, sorted_idx] = importance.sort();
        float threshold = sorted_vals[prune_count].template item<float>();

        auto prune_mask = importance < threshold; // true = prune this neuron

        for (int64_t n = 0; n < d_ff; ++n) {
            if (prune_mask[n].template item<bool>()) {
                // Zero neuron n's input weights and bias in fc1
                block.fc1->weight[n].zero_();
                if (block.fc1->bias.defined())
                    block.fc1->bias[n].zero_();
                // Zero neuron n's output connections in fc2
                // fc2.weight shape: [d_model, d_ff] — column n = neuron n's outputs
                block.fc2->weight.select(1, n).zero_();
            }
        }
        return static_cast<int>(prune_count);
    }
};

// ---------------------------------------------------------------------------
// Strategy 3: Layer Pruning
//
// Recent work (Fan et al., 2019; Sajjad et al., 2023) shows that entire
// transformer layers can be removed with surprisingly small accuracy loss,
// especially in over-parameterized models.
//
// Importance metric: mean absolute difference in model output when the
// layer is skipped vs. included. Small difference = layer is redundant.
//
// Pruning approach: zero all attention and FFN weights in the target layer.
// The LayerNorm parameters are preserved so the residual path (x + 0 = x)
// still normalizes correctly, making the block act as an identity function.
// This avoids needing to rebuild the ModuleList.
// ---------------------------------------------------------------------------
class LayerPruner {
public:
    // Score each layer by how much the model output changes when it's skipped
    static std::vector<float> compute_layer_importance(
        TransformerClassifier& model, torch::Tensor data
    ) {
        torch::NoGradGuard no_grad;
        int n = static_cast<int>(model.layers->size());
        auto full_output = model.forward(data);
        std::vector<float> importance(n);

        for (int i = 0; i < n; ++i) {
            auto skip_output = model.forward_skip_layer(data, i);
            // Mean absolute output change — larger = more important
            importance[i] = (full_output - skip_output).abs().mean().template item<float>();
        }
        return importance;
    }

    // Zero the attention + FFN weights of the least important non-pruned layer
    // LayerNorm is kept intact so residual connections pass input through
    static int prune_least_important_layer(
        TransformerClassifier& model, torch::Tensor data,
        std::vector<bool>& pruned_layers
    ) {
        auto importance = compute_layer_importance(model, data);

        // Find least important among layers not yet pruned
        int min_idx = -1;
        float min_val = std::numeric_limits<float>::max();
        for (size_t i = 0; i < importance.size(); ++i) {
            if (!pruned_layers[i] && importance[i] < min_val) {
                min_val = importance[i];
                min_idx = static_cast<int>(i);
            }
        }

        // Zero attention weights (in_proj + out_proj) and FFN weights (fc1 + fc2)
        // but keep LayerNorm so the residual path x + 0 normalizes correctly
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
// Main: train a transformer, then demonstrate all three pruning strategies
// independently and as a combined pipeline
// ---------------------------------------------------------------------------
int main() {
    torch::manual_seed(42);

    // Model configuration
    int num_layers = 4, d_model = 64, nhead = 8, d_ff = 128;
    int num_classes = 5, num_samples = 200;

    // Synthetic classification dataset
    auto data = torch::randn({num_samples, d_model});
    auto labels = torch::randint(0, num_classes, {num_samples});

    auto model = std::make_shared<TransformerClassifier>(
        num_layers, d_model, nhead, d_ff, num_classes);

    // Train the model to establish a baseline
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

    // Save trained weights so we can restore before each independent experiment
    std::vector<torch::Tensor> original_weights;
    for (const auto& p : model->parameters())
        original_weights.push_back(p.data().clone());

    auto restore_weights = [&]() {
        size_t idx = 0;
        for (auto& p : model->parameters())
            p.data().copy_(original_weights[idx++]);
    };

    // -----------------------------------------------------------------------
    // Experiment 1: Attention Head Pruning
    // Prune 2, 4, 6 out of 8 heads in every layer
    // -----------------------------------------------------------------------
    std::cout << "\n=== Attention Head Pruning ===" << std::endl;
    for (int heads_to_prune : {2, 4, 6}) {
        restore_weights(); // start from unpruned model each time
        std::cout << "\nPruning " << heads_to_prune << "/" << nhead << " heads per layer:";
        for (size_t l = 0; l < model->layers->size(); ++l) {
            auto& block = *model->layers->ptr<TransformerBlock>(l);
            auto pruned = AttentionHeadPruner::prune_heads(block, heads_to_prune);
            // Print which heads were pruned in this layer
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
    // Experiment 2: FFN Neuron Pruning
    // Prune 25%, 50%, 75% of intermediate neurons in every layer's FFN
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
    // Experiment 3: Layer Pruning
    // Iteratively remove the least important layer until only 1 remains
    // -----------------------------------------------------------------------
    std::cout << "\n=== Layer Pruning ===" << std::endl;

    // Use a separate model instance for layer pruning since it modifies structure
    auto layer_model = std::make_shared<TransformerClassifier>(
        num_layers, d_model, nhead, d_ff, num_classes);
    {
        auto dst = layer_model->parameters();
        for (size_t i = 0; i < dst.size(); ++i)
            dst[i].data().copy_(original_weights[i]);
    }

    // Display per-layer importance before pruning
    auto layer_importance = LayerPruner::compute_layer_importance(*layer_model, data);
    std::cout << "Layer importance scores:" << std::endl;
    for (size_t i = 0; i < layer_importance.size(); ++i)
        std::cout << "  Layer " << i << ": " << layer_importance[i] << std::endl;

    // Iteratively zero the least important layer
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
    // Experiment 4: Combined Pruning Pipeline
    //
    // Apply all three strategies sequentially to maximize compression:
    //   Step 1: Remove 1 layer          (coarsest granularity)
    //   Step 2: Prune 4/8 heads/layer   (medium granularity)
    //   Step 3: Prune 50% FFN neurons   (finest granularity)
    //
    // Ordering from coarse to fine avoids pruning weights in a layer
    // that will be entirely removed later.
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

    // Step 1: Layer pruning — remove the least important layer
    std::vector<bool> combined_pruned(num_layers, false);
    int removed = LayerPruner::prune_least_important_layer(*combined, data, combined_pruned);
    std::cout << "After pruning layer " << removed << ": "
              << evaluate(*combined, data, labels) << "%" << std::endl;

    // Step 2: Head pruning — remove 4 of 8 heads in each remaining layer
    for (size_t l = 0; l < combined->layers->size(); ++l)
        AttentionHeadPruner::prune_heads(*combined->layers->ptr<TransformerBlock>(l), 4);
    std::cout << "After pruning 4/8 heads: "
              << evaluate(*combined, data, labels) << "%" << std::endl;

    // Step 3: FFN pruning — remove 50% of neurons in each remaining layer
    for (size_t l = 0; l < combined->layers->size(); ++l)
        FFNPruner::prune_neurons(*combined->layers->ptr<TransformerBlock>(l), 0.5f);
    float final_sp = measure_sparsity(*combined);
    float final_acc = evaluate(*combined, data, labels);
    std::cout << "After pruning 50% FFN neurons: " << final_acc << "%"
              << " | Final sparsity: " << final_sp * 100 << "%" << std::endl;

    return 0;
}
