// Post-Training Quantization for Transformers
//
// Demonstrates layer-wise post-training quantization (PTQ) on a small
// transformer classifier. After training in FP32, each Linear layer's
// weight matrix is independently calibrated and fake-quantized to uint8
// using asymmetric quantization. The program then compares:
//   - Baseline FP32 accuracy
//   - Post-quantization accuracy
//   - Per-layer quantization error
//   - Compression ratio (FP32 -> UINT8)
//
// The TransformerQuantizer class stores a separate AsymmetricQuantizer
// per named parameter so that each layer keeps its own scale/zero_point,
// which is the standard approach for PTQ in production frameworks.
//
// Dependencies: LibTorch
// Compile: g++ -std=c++17 post_training_quantization_transformer.cpp -o post_training_quantization_transformer
//          -I$LIBTORCH_PATH/include -I$LIBTORCH_PATH/include/torch/csrc/api/include
//          -L$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$LIBTORCH_PATH/lib

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <cmath>

// ---------------------------------------------------------------------------
// Asymmetric uint8 quantizer (per-tensor)
// ---------------------------------------------------------------------------

struct AsymmetricQuantizer {
    float scale = 1.0f;
    int zero_point = 0;

    void calibrate(const torch::Tensor& tensor) {
        float min_val = tensor.min().item<float>();
        float max_val = tensor.max().item<float>();
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
// Minimal transformer encoder
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

struct Transformer : torch::nn::Module {
    torch::nn::Embedding embed{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::Linear head{nullptr};

    Transformer(int vocab_size, int d_model, int nhead, int d_ff,
                int num_layers, int num_classes) {
        embed = register_module("embed", torch::nn::Embedding(vocab_size, d_model));
        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int i = 0; i < num_layers; ++i)
            blocks->push_back(std::make_shared<TransformerBlock>(d_model, nhead, d_ff));
        head = register_module("head", torch::nn::Linear(d_model, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto h = embed->forward(x).transpose(0, 1);
        for (auto& block : *blocks)
            h = block->as<TransformerBlock>()->forward(h);
        return head->forward(h.mean(0));
    }
};

// ---------------------------------------------------------------------------
// Layer-wise post-training quantizer for transformer models
// ---------------------------------------------------------------------------

class TransformerQuantizer {
private:
    std::map<std::string, AsymmetricQuantizer> layer_quantizers;

public:
    // Calibrate a quantizer for each 2-D weight parameter (Linear layers)
    void calibrate(std::shared_ptr<Transformer> model,
                   const torch::Tensor& calibration_data) {
        torch::NoGradGuard no_grad;
        // Run a forward pass so batch-norm / layer-norm stats are populated
        model->forward(calibration_data);

        for (const auto& named_param : model->named_parameters()) {
            if (named_param.value().dim() == 2) {
                AsymmetricQuantizer quantizer;
                quantizer.calibrate(named_param.value());
                layer_quantizers[named_param.key()] = quantizer;
            }
        }
    }

    // Fake-quantize all calibrated weight parameters in-place
    void quantize_model(std::shared_ptr<Transformer> model) {
        torch::NoGradGuard no_grad;
        for (auto& named_param : model->named_parameters()) {
            auto it = layer_quantizers.find(named_param.key());
            if (it != layer_quantizers.end()) {
                auto& quantizer = it->second;
                auto quantized = quantizer.quantize(named_param.value());
                named_param.value().data().copy_(quantizer.dequantize(quantized));
            }
        }
    }

    // Report per-layer quantization error vs original weights
    void report_layer_errors(std::shared_ptr<Transformer> model,
                             const std::map<std::string, torch::Tensor>& originals) {
        std::cout << "\nPer-layer quantization error:" << std::endl;
        for (const auto& named_param : model->named_parameters()) {
            auto it = originals.find(named_param.key());
            if (it != originals.end() && named_param.value().dim() == 2) {
                float err = (named_param.value() - it->second).abs().mean().item<float>();
                std::cout << "  " << named_param.key() << ": " << err << std::endl;
            }
        }
    }

    // Compression ratio: FP32 bits / UINT8 bits for quantized layers
    float measure_compression_ratio(std::shared_ptr<Transformer> model) {
        int64_t original_bits = 0;
        int64_t quantized_bits = 0;
        for (const auto& named_param : model->named_parameters()) {
            int64_t numel = named_param.value().numel();
            if (layer_quantizers.count(named_param.key())) {
                original_bits += numel * 32;
                quantized_bits += numel * 8;
            } else {
                original_bits += numel * 32;
                quantized_bits += numel * 32; // non-quantized stays FP32
            }
        }
        return static_cast<float>(original_bits) / quantized_bits;
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

float evaluate(Transformer& model, torch::Tensor data, torch::Tensor labels) {
    torch::NoGradGuard no_grad;
    auto preds = model.forward(data).argmax(1);
    return preds.eq(labels).sum().item<float>() / data.size(0) * 100;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    torch::manual_seed(42);

    int vocab_size = 100, d_model = 64, nhead = 4, d_ff = 128;
    int num_layers = 2, num_classes = 5;
    int seq_len = 16, num_samples = 200;

    auto data = torch::randint(0, vocab_size, {num_samples, seq_len});
    auto labels = torch::randint(0, num_classes, {num_samples});

    auto model = std::make_shared<Transformer>(
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

    float baseline = evaluate(*model, data, labels);
    std::cout << "\nBaseline accuracy: " << baseline << "%" << std::endl;

    // Save original weights
    std::map<std::string, torch::Tensor> originals;
    for (const auto& np : model->named_parameters())
        originals[np.key()] = np.value().data().clone();

    // Calibrate and quantize
    TransformerQuantizer quantizer;
    quantizer.calibrate(model, data);
    quantizer.quantize_model(model);

    float quantized_acc = evaluate(*model, data, labels);
    float ratio = quantizer.measure_compression_ratio(model);

    std::cout << "\nPost-training quantization (uint8):"
              << "\n  Accuracy:          " << quantized_acc << "%"
              << "\n  Accuracy drop:     " << baseline - quantized_acc << "%"
              << "\n  Compression ratio: " << ratio << "x" << std::endl;

    quantizer.report_layer_errors(model, originals);

    // Size summary
    int64_t total_params = 0, linear_params = 0;
    for (const auto& p : model->parameters()) {
        total_params += p.numel();
        if (p.dim() == 2) linear_params += p.numel();
    }
    std::cout << "\nSize comparison:"
              << "\n  FP32:  " << total_params * 4 << " bytes"
              << "\n  UINT8 (linear weights): " << linear_params * 1
              << " bytes (4x reduction on " << linear_params << " params)"
              << std::endl;

    return 0;
}
