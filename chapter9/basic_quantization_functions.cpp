// Basic Quantization Functions
//
// Demonstrates two fundamental post-training quantization (PTQ) schemes:
//
// 1. Asymmetric quantization (uint8)
//    - Maps the full weight range [min, max] to [0, 255]
//    - Uses a zero-point offset so that float 0.0 maps to an integer value
//    - Formula: q = clamp(round(x / scale) + zero_point, 0, 255)
//    - Better for distributions not centered around zero (e.g. ReLU outputs)
//
// 2. Symmetric quantization (int8)
//    - Maps [-max_abs, +max_abs] to [-127, +127], zero maps exactly to 0
//    - No zero-point needed, simplifying hardware implementation
//    - Formula: q = clamp(round(x / scale), -127, 127)
//    - Better for weight tensors that are roughly symmetric around zero
//
// Both use per-tensor calibration: a single scale (and zero_point) per parameter
// tensor. Per-channel variants exist but are not shown here.
//
// The program trains a small network, then applies each scheme independently
// and compares accuracy degradation and quantization error vs. FP32 baseline.
//
// Dependencies: LibTorch
// Compile: g++ -std=c++17 basic_quantization_functions.cpp -o basic_quantization_functions
//          -I$LIBTORCH_PATH/include -I$LIBTORCH_PATH/include/torch/csrc/api/include
//          -L$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$LIBTORCH_PATH/lib

#include <torch/torch.h>
#include <iostream>
#include <cmath>

// Simple 3-layer feedforward network used as the quantization target
struct SimpleNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    SimpleNet(int input_dim, int hidden_dim, int output_dim) {
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, output_dim));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};

// Asymmetric quantization: maps [min, max] -> [0, 255] (uint8)
//
// Calibration computes:
//   scale      = (max - min) / 255
//   zero_point = round(-min / scale)   (the integer that represents float 0.0)
//
// Quantize:   q = clamp(round(x / scale) + zero_point, 0, 255)
// Dequantize: x_hat = scale * (q - zero_point)
//
// Quantization error is bounded by scale/2 per element.
struct AsymmetricQuantizer {
    float scale;
    int zero_point;

    // Compute scale and zero_point from the tensor's min/max range
    void calibrate(const torch::Tensor& weights) {
        float min_val = weights.min().item<float>();
        float max_val = weights.max().item<float>();
        scale = (max_val - min_val) / 255.0f;
        zero_point = static_cast<int>(-std::round(min_val / scale));
    }

    // Map float values to uint8 using affine transform
    torch::Tensor quantize(const torch::Tensor& input) {
        auto quantized = torch::clamp(
            torch::round(input / scale) + zero_point, 0, 255
        );
        return quantized.to(torch::kUInt8);
    }

    // Reconstruct approximate float values from uint8
    torch::Tensor dequantize(const torch::Tensor& quantized) {
        return scale * (quantized.to(torch::kFloat32) - zero_point);
    }
};

// Symmetric quantization: maps [-max_abs, max_abs] -> [-127, 127] (int8)
//
// Calibration computes:
//   scale = max(|weights|) / 127
//
// Quantize:   q = clamp(round(x / scale), -127, 127)
// Dequantize: x_hat = scale * q
//
// No zero_point needed — float 0.0 always maps to integer 0.
// This simplifies multiply-accumulate in hardware (no offset subtraction).
struct SymmetricQuantizer {
    float scale;

    // Compute scale from the tensor's maximum absolute value
    void calibrate(const torch::Tensor& weights) {
        float max_abs = weights.abs().max().item<float>();
        scale = max_abs / 127.0f;
    }

    // Map float values to int8 using symmetric scaling
    torch::Tensor quantize(const torch::Tensor& input) {
        auto quantized = torch::clamp(
            torch::round(input / scale), -127, 127
        );
        return quantized.to(torch::kInt8);
    }

    // Reconstruct approximate float values from int8
    torch::Tensor dequantize(const torch::Tensor& quantized) {
        return scale * quantized.to(torch::kFloat32);
    }
};

// Simulate post-training quantization on all model parameters.
// For each parameter: calibrate -> quantize -> dequantize in-place.
// After this, weights are still FP32 but restricted to the discrete set of
// values representable by the quantization scheme ("fake quantization").
template <typename Q>
void quantize_model(SimpleNet& model, Q& quantizer) {
    torch::NoGradGuard no_grad;
    for (auto& param : model.parameters()) {
        quantizer.calibrate(param.data());
        auto q = quantizer.quantize(param.data());
        param.data().copy_(quantizer.dequantize(q));
    }
}

// Compute classification accuracy (%) on the given data
float evaluate(SimpleNet& model, torch::Tensor data, torch::Tensor labels) {
    torch::NoGradGuard no_grad;
    auto preds = model.forward(data).argmax(1);
    return preds.eq(labels).sum().item<float>() / data.size(0) * 100;
}

int main() {
    torch::manual_seed(42);

    int input_dim = 20, hidden_dim = 64, num_classes = 5, num_samples = 200;
    auto data = torch::randn({num_samples, input_dim});
    auto labels = torch::randint(0, num_classes, {num_samples});

    auto model = std::make_shared<SimpleNet>(input_dim, hidden_dim, num_classes);

    // Train briefly
    torch::optim::Adam optimizer(model->parameters(), 0.001);
    std::cout << "Training model..." << std::endl;
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

    float baseline_acc = evaluate(*model, data, labels);
    std::cout << "\nBaseline accuracy: " << baseline_acc << "%" << std::endl;

    // --- Asymmetric quantization (uint8) ---
    // Restore original weights, apply asymmetric quantization, measure impact
    {
        // Restore original weights
        size_t idx = 0;
        for (auto& p : model->parameters())
            p.data().copy_(original_weights[idx++]);

        AsymmetricQuantizer aq;
        quantize_model(*model, aq);

        float acc = evaluate(*model, data, labels);

        // Measure quantization error
        float total_error = 0;
        idx = 0;
        for (const auto& p : model->parameters()) {
            total_error += (p.data() - original_weights[idx++]).abs().mean().item<float>();
        }

        std::cout << "\nAsymmetric (uint8) quantization:"
                  << "\n  Accuracy: " << acc << "%"
                  << "\n  Mean weight error: " << total_error / original_weights.size()
                  << std::endl;
    }

    // --- Symmetric quantization (int8) ---
    // Restore original weights, apply symmetric quantization, measure impact
    {
        size_t idx = 0;
        for (auto& p : model->parameters())
            p.data().copy_(original_weights[idx++]);

        SymmetricQuantizer sq;
        quantize_model(*model, sq);

        float acc = evaluate(*model, data, labels);

        float total_error = 0;
        idx = 0;
        for (const auto& p : model->parameters()) {
            total_error += (p.data() - original_weights[idx++]).abs().mean().item<float>();
        }

        std::cout << "\nSymmetric (int8) quantization:"
                  << "\n  Accuracy: " << acc << "%"
                  << "\n  Mean weight error: " << total_error / original_weights.size()
                  << std::endl;
    }

    // --- Size comparison ---
    // FP32 uses 4 bytes per weight; INT8/UINT8 use 1 byte (4x memory reduction)
    int64_t total_params = 0;
    for (const auto& p : model->parameters()) total_params += p.numel();
    std::cout << "\nSize comparison:"
              << "\n  FP32:  " << total_params * 4 << " bytes"
              << "\n  INT8:  " << total_params * 1 << " bytes (4x reduction)"
              << "\n  UINT8: " << total_params * 1 << " bytes (4x reduction)"
              << std::endl;

    return 0;
}
