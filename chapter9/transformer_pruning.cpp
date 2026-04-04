#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// Single transformer encoder block with multi-head self-attention + FFN
struct TransformerBlock : torch::nn::Module {
    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    int d_model, nhead;

    TransformerBlock(int d_model_, int nhead_, int d_ff)
        : d_model(d_model_), nhead(nhead_) {
        attn = register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model_, nhead_)));
        norm1 = register_module("norm1", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d_model_})));
        norm2 = register_module("norm2", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d_model_})));
        fc1 = register_module("fc1", torch::nn::Linear(d_model_, d_ff));
        fc2 = register_module("fc2", torch::nn::Linear(d_ff, d_model_));
    }

    // Forward with optional attention weight capture
    std::pair<torch::Tensor, torch::Tensor> forward_with_attn(torch::Tensor x) {
        auto [attn_out, attn_weights] = attn->forward(x, x, x,
            /*key_padding_mask=*/{}, /*need_weights=*/true);
        x = norm1->forward(x + attn_out);
        auto ff_out = fc2->forward(torch::relu(fc1->forward(x)));
        return {norm2->forward(x + ff_out), attn_weights};
    }

    torch::Tensor forward(torch::Tensor x) {
        return forward_with_attn(x).first;
    }
};

// Small transformer classifier for demonstration
struct TransformerClassifier : torch::nn::Module {
    std::shared_ptr<TransformerBlock> block{nullptr};
    torch::nn::Linear classifier{nullptr};
    int d_model;

    TransformerClassifier(int d_model_, int nhead, int d_ff, int num_classes)
        : d_model(d_model_) {
        block = register_module("block",
            std::make_shared<TransformerBlock>(d_model_, nhead, d_ff));
        classifier = register_module("classifier",
            torch::nn::Linear(d_model_, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [batch, d_model] -> treat as seq_len=1: [1, batch, d_model]
        x = x.unsqueeze(0);
        x = block->forward(x);
        x = x.squeeze(0); // [batch, d_model]
        return classifier->forward(x);
    }

    // Forward returning attention weights for importance computation
    std::pair<torch::Tensor, torch::Tensor> forward_with_attn(torch::Tensor x) {
        x = x.unsqueeze(0);
        auto [out, attn_w] = block->forward_with_attn(x);
        out = out.squeeze(0);
        return std::make_pair(classifier->forward(out), attn_w);
    }
};

class AttentionHeadPruner {
public:
    // Compute importance of each attention head using attention entropy
    // Lower entropy = more peaked attention = more important head
    static std::vector<float> compute_head_importance(
        TransformerClassifier& model,
        torch::Tensor data
    ) {
        torch::NoGradGuard no_grad;
        int num_heads = model.block->nhead;
        std::vector<float> head_importance(num_heads, 0.0f);

        auto [output, attn_weights] = model.forward_with_attn(data);
        // attn_weights: [batch, tgt_len, src_len] (averaged over heads by default)
        // We need per-head weights. LibTorch's average_attn_weights defaults to true.
        // Instead, compute importance from the Q/K/V projection weight magnitudes per head.

        // Use weight-based importance: L2 norm of each head's Q,K,V projection slice
        auto d_model = model.d_model;
        int d_k = d_model / num_heads;

        // in_proj_weight: [3*d_model, d_model] contains W_q, W_k, W_v stacked
        auto in_proj = model.block->attn->in_proj_weight;

        for (int h = 0; h < num_heads; ++h) {
            int start = h * d_k;
            int end = start + d_k;
            // Q slice
            float q_norm = in_proj.slice(0, start, end).norm().template item<float>();
            // K slice
            float k_norm = in_proj.slice(0, d_model + start, d_model + end).norm().template item<float>();
            // V slice
            float v_norm = in_proj.slice(0, 2 * d_model + start, 2 * d_model + end).norm().template item<float>();

            head_importance[h] = q_norm + k_norm + v_norm;
        }

        return head_importance;
    }

    // Prune the least important heads by zeroing their Q/K/V projection weights
    static std::vector<int> prune_heads(
        TransformerClassifier& model,
        int num_heads_to_prune
    ) {
        auto importance = compute_head_importance(model, torch::randn({1, model.d_model}));
        int num_heads = model.block->nhead;
        int d_model = model.d_model;
        int d_k = d_model / num_heads;

        // Sort heads by importance (ascending) to find least important
        std::vector<int> head_indices(num_heads);
        std::iota(head_indices.begin(), head_indices.end(), 0);
        std::sort(head_indices.begin(), head_indices.end(),
            [&](int a, int b) { return importance[a] < importance[b]; });

        std::vector<int> pruned_heads(
            head_indices.begin(),
            head_indices.begin() + num_heads_to_prune);

        // Zero out Q/K/V projection weights and biases for pruned heads
        torch::NoGradGuard no_grad;
        auto& in_proj_w = model.block->attn->in_proj_weight;
        auto& in_proj_b = model.block->attn->in_proj_bias;
        auto& out_proj_w = model.block->attn->out_proj->weight;
        auto& out_proj_b = model.block->attn->out_proj->bias;

        for (int h : pruned_heads) {
            int start = h * d_k;
            int end = start + d_k;

            // Zero Q, K, V rows in in_proj_weight [3*d_model, d_model]
            in_proj_w.slice(0, start, end).zero_();
            in_proj_w.slice(0, d_model + start, d_model + end).zero_();
            in_proj_w.slice(0, 2 * d_model + start, 2 * d_model + end).zero_();

            // Zero corresponding biases
            if (in_proj_b.defined()) {
                in_proj_b.slice(0, start, end).zero_();
                in_proj_b.slice(0, d_model + start, d_model + end).zero_();
                in_proj_b.slice(0, 2 * d_model + start, 2 * d_model + end).zero_();
            }

            // Zero output projection columns for this head
            out_proj_w.slice(1, start, end).zero_();
        }

        return pruned_heads;
    }

    // Measure sparsity of the entire model
    static float measure_sparsity(TransformerClassifier& model) {
        int64_t total = 0, zeros = 0;
        for (const auto& p : model.parameters()) {
            total += p.numel();
            zeros += (p.data() == 0).sum().item<int64_t>();
        }
        return static_cast<float>(zeros) / total;
    }
};

int main() {
    torch::manual_seed(42);

    int d_model = 64, nhead = 8, d_ff = 128, num_classes = 5, num_samples = 200;
    auto data = torch::randn({num_samples, d_model});
    auto labels = torch::randint(0, num_classes, {num_samples});

    auto model = std::make_shared<TransformerClassifier>(d_model, nhead, d_ff, num_classes);

    // Train briefly
    torch::optim::Adam optimizer(model->parameters(), 0.001);
    std::cout << "Training transformer model..." << std::endl;
    for (int epoch = 0; epoch < 50; ++epoch) {
        optimizer.zero_grad();
        auto loss = torch::nn::functional::cross_entropy(model->forward(data), labels);
        loss.backward();
        optimizer.step();
        if (epoch % 10 == 0)
            std::cout << "  Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }

    // Save original weights
    std::vector<torch::Tensor> original_weights;
    for (const auto& p : model->parameters())
        original_weights.push_back(p.data().clone());

    // Evaluate before pruning
    {
        torch::NoGradGuard no_grad;
        auto preds = model->forward(data).argmax(1);
        float acc = preds.eq(labels).sum().item<float>() / num_samples * 100;
        std::cout << "\nBefore pruning - Accuracy: " << acc << "%" << std::endl;
    }

    int64_t total_params = 0;
    for (const auto& p : model->parameters()) total_params += p.numel();
    std::cout << "Total parameters: " << total_params << std::endl;

    // Compute and display head importance
    auto importance = AttentionHeadPruner::compute_head_importance(*model, data);
    std::cout << "\nHead importance scores:" << std::endl;
    for (int h = 0; h < nhead; ++h)
        std::cout << "  Head " << h << ": " << importance[h] << std::endl;

    // Prune increasing numbers of heads
    std::vector<int> heads_to_prune_counts = {1, 2, 4, 6};

    for (int num_prune : heads_to_prune_counts) {
        // Restore original weights
        size_t idx = 0;
        for (auto& p : model->parameters())
            p.data().copy_(original_weights[idx++]);

        auto pruned = AttentionHeadPruner::prune_heads(*model, num_prune);
        float sparsity = AttentionHeadPruner::measure_sparsity(*model);

        torch::NoGradGuard no_grad;
        auto preds = model->forward(data).argmax(1);
        float acc = preds.eq(labels).sum().item<float>() / num_samples * 100;

        std::cout << "\nPruned " << num_prune << "/" << nhead << " heads (";
        for (size_t i = 0; i < pruned.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << pruned[i];
        }
        std::cout << ") | Sparsity: " << sparsity * 100 << "%"
                  << " | Accuracy: " << acc << "%" << std::endl;
    }

    return 0;
}
