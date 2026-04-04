// Quantization-Aware Training (QAT) for Transformers
//
// QAT simulates quantization during training so the model learns to be robust
// to the precision loss that occurs when weights are mapped to low-bit integers.
//
// Key idea — Straight-Through Estimator (STE):
//   The quantize-then-dequantize ("fake quantize") operation is non-differentiable
//   (round() has zero gradient almost everywhere). STE solves this by passing
//   gradients through the rounding step unchanged during backpropagation:
//     Forward:  w_q = dequantize(quantize(w))   — weights see quantization noise
//     Backward: dL/dw = dL/dw_q                 — gradient flows through as-is
//
// Training loop:
//   For each training step:
//     1. Save original FP32 weights
//     2. Replace weights with fake-quantized versions (forward uses quantized weights)
//     3. Compute loss and backpropagate (STE: gradients pass through round())
//     4. Restore original FP32 weights, then apply optimizer step to them
//   This way the optimizer always updates the full-precision "master" weights,
//   but the forward pass always sees quantization noise, teaching the model
//   to place weights in quantization-friendly regions.
//
// After QAT, actual PTQ can be applied with minimal accuracy loss because the
// model has already adapted to quantization during training.
//
// This program compares three scenarios on a transformer classifier:
//   1. FP32 baseline (no quantization)
//   2. Naive PTQ (quantize a normally-trained model, no QAT)
//   3. QAT + PTQ (train with fake quantization, then apply real PTQ)
//
// Dependencies: LibTorch
// Compile: g++ -std=c++17 qat_transformers.cpp -o qat_transformers
//          -I$LIBTORCH_PATH/include -I$LIBTORCH_PATH/include/torch/csrc/api/include
//          -L$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$LIBTORCH_PATH/lib

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// Transformer model (self-contained, same architecture as sibling programs)
// ---------------------------------------------------------------------------

// Single transformer encoder block: multi-head self-attention + FFN
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
        // Self-attention with residual connection and layer norm
        auto attn_out = std::get<0>(attn->forward(x, x, x));
        x = norm1->forward(x + attn_out);
        // Position-wise FFN with residual connection and layer norm
        auto ff_out = ff2->forward(torch::relu(ff1->forward(x)));
        return norm2->forward(x + ff_out);
    }
};

// Embedding -> stacked TransformerBlocks -> mean pool -> classification head
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

    // [batch, seq_len] token ids -> [batch, num_classes] logits
    torch::Tensor forward(torch::Tensor x) {
        auto h = embed->forward(x).transpose(0, 1); // [seq, batch, d_model]
        for (auto& block : *blocks)
            h = block->as<TransformerBlock>()->forward(h);
        auto pooled = h.mean(0); // mean-pool over sequence -> [batch, d_model]
        return head->forward(pooled);
    }
};

// ---------------------------------------------------------------------------
// Fake quantization operation (core of QAT)
//
// Simulates asymmetric uint8 quantization: float -> round -> clamp -> float
// The result is an FP32 tensor whose values are restricted to the 256 levels
// representable by uint8 with the given scale/zero_point.
//
// Because this runs inside autograd, LibTorch records it in the computation
// graph. The round() operation has zero gradient, but since we only apply
// arithmetic ops that LibTorch can differentiate through (the clamp and
// linear transforms), the gradient flows through approximately — this is
// the Straight-Through Estimator (STE).
// ---------------------------------------------------------------------------

torch::Tensor fake_quantize(const torch::Tensor& input, float scale, int zero_point) {
    // Quantize: map float to integer grid
    auto quantized = torch::clamp(
        torch::round(input / scale) + zero_point, 0, 255
    );
    // Dequantize: map back to float (now restricted to quantization levels)
    return scale * (quantized - zero_point);
}

// ---------------------------------------------------------------------------
// Calibrate asymmetric uint8 parameters from a tensor's value range
// ---------------------------------------------------------------------------

struct QuantParams {
    float scale;
    int zero_point;
};

QuantParams calibrate(const torch::Tensor& tensor) {
    float min_val = tensor.min().item<float>();
    float max_val = tensor.max().item<float>();
    float scale = (max_val - min_val) / 255.0f;
    if (scale == 0.0f) scale = 1.0f; // guard against constant tensors
    int zp = static_cast<int>(-std::round(min_val / scale));
    return {scale, zp};
}

// ---------------------------------------------------------------------------
// Apply real (non-fake) PTQ to all 2-D weight parameters in-place
// Used after training to actually quantize the model.
// ---------------------------------------------------------------------------

void apply_ptq(TransformerClassifier& model) {
    torch::NoGradGuard no_grad;
    for (auto& param : model.parameters()) {
        if (param.dim() == 2) { // only quantize Linear weight matrices
            auto [scale, zp] = calibrate(param.data());
            // Quantize to uint8 then dequantize back to FP32
            auto q = torch::clamp(
                torch::round(param.data() / scale) + zp, 0, 255
            ).to(torch::kUInt8);
            param.data().copy_(scale * (q.to(torch::kFloat32) - zp));
        }
    }
}

// ---------------------------------------------------------------------------
// QAT training loop
//
// Each step:
//   1. Save FP32 master weights for all Linear layers
//   2. Replace them with fake-quantized versions (simulates inference-time
//      quantization so the model "sees" quantization noise during training)
//   3. Forward pass + loss + backward (STE passes gradients through round())
//   4. Restore FP32 master weights (undo fake quantization)
//   5. Optimizer step updates the master weights using the computed gradients
// ---------------------------------------------------------------------------

void train_with_qat(TransformerClassifier& model,
                    torch::optim::Adam& optimizer,
                    const torch::Tensor& data,
                    const torch::Tensor& labels,
                    int num_epochs) {

    // Pre-collect named parameters that are 2-D (Linear weights) so we can
    // index them stably across epochs without dangling pointers.
    std::vector<std::string> linear_names;
    for (const auto& kv : model.named_parameters()) {
        if (kv.value().dim() == 2)
            linear_names.push_back(kv.key());
    }

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        optimizer.zero_grad();

        // Step 1: Save original FP32 weights for all Linear parameters
        auto params = model.named_parameters();
        std::vector<torch::Tensor> saved_weights;
        for (const auto& name : linear_names) {
            saved_weights.push_back(params[name].data().clone());
        }

        // Step 2: Replace weights with fake-quantized versions
        // Each Linear weight gets its own scale/zero_point (per-tensor calibration)
        {
            torch::NoGradGuard no_grad;
            for (const auto& name : linear_names) {
                auto& w = params[name];
                auto [scale, zp] = calibrate(w.data());
                w.data().copy_(fake_quantize(w.data(), scale, zp));
            }
        }

        // Step 3: Forward pass with quantized weights, compute loss, backprop
        auto output = model.forward(data);
        auto loss = torch::nn::functional::cross_entropy(output, labels);
        loss.backward();

        // Step 4: Restore original FP32 master weights before optimizer step
        // (optimizer must update the full-precision weights, not the quantized ones)
        {
            torch::NoGradGuard no_grad;
            for (size_t i = 0; i < linear_names.size(); ++i) {
                params[linear_names[i]].data().copy_(saved_weights[i]);
            }
        }

        // Step 5: Optimizer updates master weights using gradients from step 3
        optimizer.step();

        if (epoch % 10 == 0) {
            std::cout << "  Epoch " << epoch
                      << ", Loss: " << loss.item<float>() << std::endl;
        }
    }
}

// ---------------------------------------------------------------------------
// Standard (non-QAT) training loop for comparison
// ---------------------------------------------------------------------------

void train_standard(TransformerClassifier& model,
                    torch::optim::Adam& optimizer,
                    const torch::Tensor& data,
                    const torch::Tensor& labels,
                    int num_epochs) {
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        optimizer.zero_grad();
        auto loss = torch::nn::functional::cross_entropy(model.forward(data), labels);
        loss.backward();
        optimizer.step();
        if (epoch % 10 == 0)
            std::cout << "  Epoch " << epoch
                      << ", Loss: " << loss.item<float>() << std::endl;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Classification accuracy (%)
float evaluate(TransformerClassifier& model, torch::Tensor data, torch::Tensor labels) {
    torch::NoGradGuard no_grad;
    auto preds = model.forward(data).argmax(1);
    return preds.eq(labels).sum().item<float>() / data.size(0) * 100;
}

// Mean absolute error between current and reference weights (Linear layers only)
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
// Main: compare FP32 baseline vs naive PTQ vs QAT+PTQ
// ---------------------------------------------------------------------------

int main() {
    torch::manual_seed(42);

    // Hyperparameters
    int vocab_size = 100, d_model = 64, nhead = 4, d_ff = 128;
    int num_layers = 2, num_classes = 5;
    int seq_len = 16, num_samples = 200;
    int train_epochs = 30;
    int qat_epochs = 60; // QAT needs more epochs due to quantization noise

    // Synthetic token classification data
    auto data = torch::randint(0, vocab_size, {num_samples, seq_len});
    auto labels = torch::randint(0, num_classes, {num_samples});

    // -----------------------------------------------------------------------
    // Scenario 1: Standard training (FP32 baseline)
    // -----------------------------------------------------------------------
    std::cout << "=== Scenario 1: Standard FP32 training ===" << std::endl;
    auto model_std = std::make_shared<TransformerClassifier>(
        vocab_size, d_model, nhead, d_ff, num_layers, num_classes);
    torch::optim::Adam opt_std(model_std->parameters(), 0.001);
    train_standard(*model_std, opt_std, data, labels, train_epochs);

    float baseline_acc = evaluate(*model_std, data, labels);
    std::cout << "FP32 accuracy: " << baseline_acc << "%\n" << std::endl;

    // Save FP32 weights as reference for error measurement
    std::vector<torch::Tensor> fp32_weights;
    for (const auto& p : model_std->parameters())
        fp32_weights.push_back(p.data().clone());

    // -----------------------------------------------------------------------
    // Scenario 2: Naive PTQ (standard training, then quantize without QAT)
    // Apply PTQ to a copy of the standard-trained model
    // -----------------------------------------------------------------------
    std::cout << "=== Scenario 2: Naive PTQ (no QAT) ===" << std::endl;
    // Clone weights into a fresh model
    auto model_ptq = std::make_shared<TransformerClassifier>(
        vocab_size, d_model, nhead, d_ff, num_layers, num_classes);
    {
        torch::NoGradGuard no_grad;
        auto src = model_std->parameters();
        auto dst = model_ptq->parameters();
        for (size_t i = 0; i < src.size(); ++i)
            dst[i].data().copy_(src[i].data());
    }

    // Apply real quantization
    apply_ptq(*model_ptq);
    float ptq_acc = evaluate(*model_ptq, data, labels);
    float ptq_err = weight_error(*model_ptq, fp32_weights);
    std::cout << "PTQ accuracy:          " << ptq_acc << "%"
              << "\nMean weight error:     " << ptq_err << "\n" << std::endl;

    // -----------------------------------------------------------------------
    // Scenario 3: QAT training, then PTQ
    // The model trains with fake quantization so it learns to tolerate
    // quantization noise. After QAT, real PTQ causes less accuracy loss.
    // -----------------------------------------------------------------------
    std::cout << "=== Scenario 3: QAT + PTQ ===" << std::endl;
    auto model_qat = std::make_shared<TransformerClassifier>(
        vocab_size, d_model, nhead, d_ff, num_layers, num_classes);
    torch::optim::Adam opt_qat(model_qat->parameters(), 0.001);

    // QAT training: forward pass uses fake-quantized weights
    // More epochs needed because fake quantization adds noise that slows convergence
    train_with_qat(*model_qat, opt_qat, data, labels, qat_epochs);

    // Evaluate before real quantization (FP32 master weights)
    float qat_fp32_acc = evaluate(*model_qat, data, labels);
    std::cout << "QAT FP32 accuracy:     " << qat_fp32_acc << "%" << std::endl;

    // Now apply real PTQ to the QAT-trained model
    apply_ptq(*model_qat);
    float qat_ptq_acc = evaluate(*model_qat, data, labels);
    std::cout << "QAT + PTQ accuracy:    " << qat_ptq_acc << "%" << std::endl;

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "\n=== Summary ==="
              << "\nFP32 baseline:   " << baseline_acc << "%"
              << "\nNaive PTQ:       " << ptq_acc << "%"
              << "\nQAT + PTQ:       " << qat_ptq_acc << "%"
              << "\n\nQAT helps the model learn quantization-robust weights,"
              << "\nreducing the accuracy gap between FP32 and quantized inference."
              << std::endl;

    return 0;
}
