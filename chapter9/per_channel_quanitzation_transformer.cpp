// Per-Channel Quantization for Transformers
//
// Transformer model:
//   This program builds a TransformerClassifier — a sequence classification
//   model that maps variable-length token sequences to one of N classes.
//   Architecture:
//     1. Embedding layer: token ids -> d_model-dimensional vectors
//     2. Stacked TransformerBlocks (x num_layers), each containing:
//        - Multi-head self-attention (queries, keys, values from same input)
//        - Add & LayerNorm (residual connection)
//        - Position-wise feedforward network (Linear -> ReLU -> Linear)
//        - Add & LayerNorm (residual connection)
//     3. Mean pooling over the sequence dimension
//     4. Linear classification head -> num_classes logits
//   The model is trained on synthetic random token sequences with cross-entropy
//   loss, then post-training quantization is applied to its weight matrices.
//
// Quantization comparison:
//   After training, the program applies two post-training quantization (PTQ)
//   schemes to the Linear layer weights and compares their impact:
//
//   Per-tensor (baseline): a single scale and zero_point for the entire weight
//     matrix. Simple but loses precision when different output channels have
//     very different value ranges — common in transformer attention projections
//     (Q/K/V) and FFN layers.
//
//   Per-channel: each output channel (row) of a weight matrix gets its own
//     scale and zero_point, better preserving channels with different ranges.
//     This is the standard approach used in production quantized transformers
//     (e.g. TensorRT, ONNX Runtime) because transformer weight rows often
//     have heterogeneous distributions.
//
//   Both use asymmetric uint8 mapping: [min, max] -> [0, 255].
//   Only 2-D weight parameters (Linear layers) are quantized; biases, layer
//   norm parameters, and embeddings are left in FP32.
//
// Dependencies: LibTorch
// Compile: g++ -std=c++17 per_channel_quanitzation_transformer.cpp -o per_channel_quanitzation_transformer
//          -I$LIBTORCH_PATH/include -I$LIBTORCH_PATH/include/torch/csrc/api/include
//          -L$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$LIBTORCH_PATH/lib

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// Minimal transformer encoder (self-contained, no external headers needed)
// ---------------------------------------------------------------------------

struct TransformerBlock : torch::nn::Module {
    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Linear ff1{nullptr}, ff2{nullptr};

    TransformerBlock(int d_model, int nhead, int d_ff) {
        attn = register_module("attn", torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model, nhead)));
        norm1 = register_module("norm1", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d_model})));
        norm2 = register_module("norm2", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d_model})));
        ff1 = register_module("ff1", torch::nn::Linear(d_model, d_ff));
        ff2 = register_module("ff2", torch::nn::Linear(d_ff, d_model));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto attn_out = std::get<0>(attn->forward(x, x, x));
        x = norm1->forward(x + attn_out);
        auto ff_out = ff2->forward(torch::relu(ff1->forward(x)));
        return norm2->forward(x + ff_out);
    }
};

struct TransformerClassifier : torch::nn::Module {
    torch::nn::Embedding embed{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::Linear head{nullptr};

    TransformerClassifier(int vocab_size, int d_model, int nhead, int d_ff,
                          int num_layers, int num_classes) {
        embed = register_module("embed", torch::nn::Embedding(vocab_size, d_model));
        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int i = 0; i < num_layers; ++i)
            blocks->push_back(std::make_shared<TransformerBlock>(d_model, nhead, d_ff));
        head = register_module("head", torch::nn::Linear(d_model, num_classes));
    }

    // input: [batch, seq_len] of token ids -> [batch, num_classes] logits
    torch::Tensor forward(torch::Tensor x) {
        auto h = embed->forward(x).transpose(0, 1); // [seq, batch, d_model]
        for (auto& block : *blocks)
            h = block->as<TransformerBlock>()->forward(h);
        // Mean-pool over sequence dimension, then classify
        auto pooled = h.mean(0); // [batch, d_model]
        return head->forward(pooled);
    }
};

// ---------------------------------------------------------------------------
// Per-channel asymmetric quantizer (uint8)
// One scale + zero_point per output channel (row) of a weight matrix.
// ---------------------------------------------------------------------------

class PerChannelQuantizer {
private:
    std::vector<float> scales;
    std::vector<int> zero_points;

public:
    // Calibrate per row: weight shape [out_features, in_features]
    void calibrate_linear_layer(const torch::Tensor& weight) {
        int64_t num_channels = weight.size(0);
        scales.resize(num_channels);
        zero_points.resize(num_channels);

        for (int64_t i = 0; i < num_channels; ++i) {
            auto channel = weight[i];
            float min_val = channel.min().item<float>();
            float max_val = channel.max().item<float>();
            scales[i] = (max_val - min_val) / 255.0f;
            if (scales[i] == 0.0f) scales[i] = 1.0f; // avoid division by zero
            zero_points[i] = static_cast<int>(-std::round(min_val / scales[i]));
        }
    }

    // Quantize each row with its own scale/zero_point
    torch::Tensor quantize(const torch::Tensor& weight) {
        auto quantized = torch::empty_like(weight, torch::kUInt8);
        for (int64_t i = 0; i < weight.size(0); ++i) {
            quantized[i] = torch::clamp(
                torch::round(weight[i] / scales[i]) + zero_points[i], 0, 255
            ).to(torch::kUInt8);
        }
        return quantized;
    }

    // Dequantize back to float using per-row parameters
    torch::Tensor dequantize(const torch::Tensor& quantized) {
        auto result = torch::empty({quantized.sizes()}, torch::kFloat32);
        for (int64_t i = 0; i < quantized.size(0); ++i) {
            result[i] = scales[i] * (quantized[i].to(torch::kFloat32) - zero_points[i]);
        }
        return result;
    }
};

// ---------------------------------------------------------------------------
// Per-tensor asymmetric quantizer (uint8) — baseline for comparison
// ---------------------------------------------------------------------------

struct PerTensorQuantizer {
    float scale;
    int zero_point;

    void calibrate(const torch::Tensor& weight) {
        float min_val = weight.min().item<float>();
        float max_val = weight.max().item<float>();
        scale = (max_val - min_val) / 255.0f;
        if (scale == 0.0f) scale = 1.0f;
        zero_point = static_cast<int>(-std::round(min_val / scale));
    }

    torch::Tensor quantize(const torch::Tensor& input) {
        return torch::clamp(
            torch::round(input / scale) + zero_point, 0, 255
        ).to(torch::kUInt8);
    }

    torch::Tensor dequantize(const torch::Tensor& quantized) {
        return scale * (quantized.to(torch::kFloat32) - zero_point);
    }
};

// ---------------------------------------------------------------------------
// Apply fake-quantization to all 2-D weight parameters (Linear layers)
// ---------------------------------------------------------------------------

void apply_per_channel(TransformerClassifier& model,
                       const std::vector<torch::Tensor>& originals) {
    torch::NoGradGuard no_grad;
    size_t idx = 0;
    PerChannelQuantizer pcq;
    for (auto& param : model.parameters()) {
        param.data().copy_(originals[idx]);
        if (param.dim() == 2) { // Linear weight
            pcq.calibrate_linear_layer(param.data());
            auto q = pcq.quantize(param.data());
            param.data().copy_(pcq.dequantize(q));
        }
        ++idx;
    }
}

void apply_per_tensor(TransformerClassifier& model,
                      const std::vector<torch::Tensor>& originals) {
    torch::NoGradGuard no_grad;
    size_t idx = 0;
    PerTensorQuantizer ptq;
    for (auto& param : model.parameters()) {
        param.data().copy_(originals[idx]);
        if (param.dim() == 2) {
            ptq.calibrate(param.data());
            auto q = ptq.quantize(param.data());
            param.data().copy_(ptq.dequantize(q));
        }
        ++idx;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float evaluate(TransformerClassifier& model, torch::Tensor data, torch::Tensor labels) {
    torch::NoGradGuard no_grad;
    auto preds = model.forward(data).argmax(1);
    return preds.eq(labels).sum().item<float>() / data.size(0) * 100;
}

float weight_error(TransformerClassifier& model,
                   const std::vector<torch::Tensor>& originals) {
    float total = 0;
    int count = 0;
    size_t idx = 0;
    for (const auto& param : model.parameters()) {
        if (param.dim() == 2) {
            total += (param.data() - originals[idx]).abs().mean().item<float>();
            ++count;
        }
        ++idx;
    }
    return count > 0 ? total / count : 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    torch::manual_seed(42);

    // Model hyperparameters
    int vocab_size = 100, d_model = 64, nhead = 4, d_ff = 128;
    int num_layers = 2, num_classes = 5;
    int seq_len = 16, num_samples = 200;

    // Synthetic data: random token sequences
    auto data = torch::randint(0, vocab_size, {num_samples, seq_len});
    auto labels = torch::randint(0, num_classes, {num_samples});

    auto model = std::make_shared<TransformerClassifier>(
        vocab_size, d_model, nhead, d_ff, num_layers, num_classes);

    // Train
    torch::optim::Adam optimizer(model->parameters(), 0.001);
    std::cout << "Training transformer..." << std::endl;
    for (int epoch = 0; epoch < 30; ++epoch) {
        optimizer.zero_grad();
        auto loss = torch::nn::functional::cross_entropy(model->forward(data), labels);
        loss.backward();
        optimizer.step();
        if (epoch % 10 == 0)
            std::cout << "  Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }

    // Save original weights
    std::vector<torch::Tensor> originals;
    for (const auto& p : model->parameters())
        originals.push_back(p.data().clone());

    float baseline = evaluate(*model, data, labels);
    std::cout << "\nBaseline accuracy: " << baseline << "%" << std::endl;

    // Count linear-layer parameters
    int64_t linear_params = 0, total_params = 0;
    for (const auto& p : model->parameters()) {
        total_params += p.numel();
        if (p.dim() == 2) linear_params += p.numel();
    }
    std::cout << "Total parameters: " << total_params
              << " (linear weights: " << linear_params << ")" << std::endl;

    // --- Per-tensor quantization ---
    apply_per_tensor(*model, originals);
    float pt_acc = evaluate(*model, data, labels);
    float pt_err = weight_error(*model, originals);
    std::cout << "\nPer-tensor (uint8):"
              << "\n  Accuracy:          " << pt_acc << "%"
              << "\n  Mean weight error: " << pt_err << std::endl;

    // --- Per-channel quantization ---
    apply_per_channel(*model, originals);
    float pc_acc = evaluate(*model, data, labels);
    float pc_err = weight_error(*model, originals);
    std::cout << "\nPer-channel (uint8):"
              << "\n  Accuracy:          " << pc_acc << "%"
              << "\n  Mean weight error: " << pc_err << std::endl;

    // --- Summary ---
    std::cout << "\nSummary:"
              << "\n  Per-channel reduces quantization error by preserving per-row"
              << "\n  dynamic range, which is especially important for transformer"
              << "\n  attention and FFN layers where channels can have very different"
              << "\n  weight distributions."
              << "\n\n  FP32 size:  " << linear_params * 4 << " bytes"
              << "\n  UINT8 size: " << linear_params * 1 << " bytes (4x reduction)"
              << std::endl;

    return 0;
}
