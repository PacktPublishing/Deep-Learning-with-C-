// Basic Quantization Functions
#include <torch/torch.h>
#include <iostream>
#include <cmath>

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
struct AsymmetricQuantizer {
    float scale;
    int zero_point;

    void calibrate(const torch::Tensor& weights) {
        float min_val = weights.min().item<float>();
        float max_val = weights.max().item<float>();
        scale = (max_val - min_val) / 255.0f;
        zero_point = static_cast<int>(-std::round(min_val / scale));
    }

    torch::Tensor quantize(const torch::Tensor& input) {
        auto quantized = torch::clamp(
            torch::round(input / scale) + zero_point, 0, 255
        );
        return quantized.to(torch::kUInt8);
    }

    torch::Tensor dequantize(const torch::Tensor& quantized) {
        return scale * (quantized.to(torch::kFloat32) - zero_point);
    }
};

// Symmetric quantization: maps [-max_abs, max_abs] -> [-127, 127] (int8)
struct SymmetricQuantizer {
    float scale;

    void calibrate(const torch::Tensor& weights) {
        float max_abs = weights.abs().max().item<float>();
        scale = max_abs / 127.0f;
    }

    torch::Tensor quantize(const torch::Tensor& input) {
        auto quantized = torch::clamp(
            torch::round(input / scale), -127, 127
        );
        return quantized.to(torch::kInt8);
    }

    torch::Tensor dequantize(const torch::Tensor& quantized) {
        return scale * quantized.to(torch::kFloat32);
    }
};

// Apply quantize->dequantize to all model parameters in-place
template <typename Q>
void quantize_model(SimpleNet& model, Q& quantizer) {
    torch::NoGradGuard no_grad;
    for (auto& param : model.parameters()) {
        quantizer.calibrate(param.data());
        auto q = quantizer.quantize(param.data());
        param.data().copy_(quantizer.dequantize(q));
    }
}

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
    int64_t total_params = 0;
    for (const auto& p : model->parameters()) total_params += p.numel();
    std::cout << "\nSize comparison:"
              << "\n  FP32:  " << total_params * 4 << " bytes"
              << "\n  INT8:  " << total_params * 1 << " bytes (4x reduction)"
              << "\n  UINT8: " << total_params * 1 << " bytes (4x reduction)"
              << std::endl;

    return 0;
}
