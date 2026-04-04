// iterative_pruning.cpp
//
// Iterative magnitude pruning with finetuning recovery.
//
// Unlike one-shot pruning (magnitude_pruning.cpp), this implements the
// prune → finetune → repeat cycle that allows the surviving weights to
// adapt after each pruning step, recovering lost accuracy before the
// next round of pruning at a higher sparsity target.
//
// Pipeline per sparsity level:
//   1. Prune: compute global magnitude threshold, create binary masks,
//      zero weights below threshold
//   2. Finetune: train with a lower learning rate while re-applying
//      masks after every optimizer step (prevents pruned weights from
//      recovering through gradient updates)
//   3. Evaluate: measure accuracy and sparsity, report recovery
//
// Dependencies: LibTorch (PyTorch C++ API)
// Compile:
//   g++ -std=c++17 iterative_pruning.cpp -o iterative_pruning \
//     -I$LIBTORCH_PATH/include -I$LIBTORCH_PATH/include/torch/csrc/api/include \
//     -L$LIBTORCH_PATH/lib -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$LIBTORCH_PATH/lib

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>

// 3-layer feedforward network: input -> hidden -> hidden -> output
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

// Global magnitude pruning with persistent binary masks.
//
// Key difference from magnitude_pruning.cpp's MagnitudePruner:
//   - Stores masks so they can be re-applied after each optimizer step
//   - This prevents the optimizer from "reviving" pruned weights via
//     gradient updates during finetuning
class MagnitudePruner {
private:
    torch::nn::Module& model;
    float sparsity_target;
    std::vector<torch::Tensor> masks; // 1 = keep, 0 = pruned

public:
    MagnitudePruner(torch::nn::Module& model_, float sparsity_)
        : model(model_), sparsity_target(sparsity_) {}

    // Find the magnitude value at the sparsity_target percentile
    // using O(n) partial sort (nth_element)
    float compute_threshold() {
        std::vector<float> all_weights;
        for (const auto& param : model.parameters()) {
            auto w = param.data().abs().flatten().cpu();
            auto acc = w.accessor<float, 1>();
            for (int64_t i = 0; i < acc.size(0); ++i)
                all_weights.push_back(acc[i]);
        }
        size_t idx = static_cast<size_t>(all_weights.size() * sparsity_target);
        std::nth_element(all_weights.begin(), all_weights.begin() + idx, all_weights.end());
        return all_weights[idx];
    }

    // Compute binary masks from current weights and apply them.
    // mask[i] = 1 if |weight[i]| >= threshold, 0 otherwise.
    void apply_pruning() {
        float threshold = compute_threshold();
        masks.clear();
        for (auto& param : model.parameters()) {
            auto mask = (param.data().abs() >= threshold).to(torch::kFloat);
            masks.push_back(mask);
            param.data().mul_(mask);
        }
    }

    // Re-apply stored masks after an optimizer step.
    // The optimizer may assign nonzero gradients to pruned positions;
    // this zeros them again to maintain the sparsity pattern.
    void reapply_masks() {
        size_t i = 0;
        for (auto& param : model.parameters()) {
            param.data().mul_(masks[i]);
            ++i;
        }
    }

    // Fraction of parameters that are exactly zero
    float measure_sparsity() {
        int64_t total = 0, zeros = 0;
        for (const auto& param : model.parameters()) {
            total += param.numel();
            zeros += (param.data() == 0).sum().item<int64_t>();
        }
        return static_cast<float>(zeros) / total;
    }
};

// Classification accuracy (%) on given data/labels
float evaluate(SimpleNet& model, torch::Tensor data, torch::Tensor labels) {
    torch::NoGradGuard no_grad;
    auto preds = model.forward(data).argmax(1);
    return preds.eq(labels).sum().item<float>() / labels.size(0) * 100;
}

// Core pipeline: iterate through increasing sparsity targets.
// Each round prunes from the current (already-pruned) model, so weights
// pruned in earlier rounds stay pruned — sparsity only increases.
// Finetuning uses a lower learning rate (1e-4) than initial training (1e-3)
// to gently adjust surviving weights without large destabilizing updates.
void iterative_pruning_pipeline(
    SimpleNet& model,
    torch::Tensor data,
    torch::Tensor labels,
    const std::vector<float>& sparsity_schedule,
    int finetune_epochs
) {
    for (float target_sparsity : sparsity_schedule) {
        std::cout << "\n--- Pruning to " << target_sparsity * 100 << "% sparsity ---" << std::endl;

        // Step 1: Prune — compute new threshold on current weights and mask
        MagnitudePruner pruner(model, target_sparsity);
        pruner.apply_pruning();
        float achieved = pruner.measure_sparsity();
        std::cout << "Achieved sparsity: " << achieved * 100 << "%" << std::endl;

        float acc_before = evaluate(model, data, labels);
        std::cout << "Accuracy after pruning (before finetune): " << acc_before << "%" << std::endl;

        // Step 2: Finetune — lower LR to recover accuracy while preserving masks
        auto optimizer = torch::optim::Adam(model.parameters(),
            torch::optim::AdamOptions(1e-4));

        for (int epoch = 0; epoch < finetune_epochs; ++epoch) {
            optimizer.zero_grad();
            auto loss = torch::nn::functional::cross_entropy(model.forward(data), labels);
            loss.backward();
            optimizer.step();

            // Critical: re-apply mask after optimizer.step() so pruned
            // weights don't drift back to nonzero via gradient updates
            pruner.reapply_masks();

            if (epoch % 5 == 0 || epoch == finetune_epochs - 1)
                std::cout << "  Finetune epoch " << epoch
                          << ", Loss: " << loss.item<float>() << std::endl;
        }

        // Step 3: Evaluate — report accuracy recovery from finetuning
        float acc_after = evaluate(model, data, labels);
        float final_sparsity = pruner.measure_sparsity();
        std::cout << "Accuracy after finetune: " << acc_after << "%"
                  << " | Sparsity: " << final_sparsity * 100 << "%"
                  << " | Recovery: " << (acc_after - acc_before) << "pp" << std::endl;
    }
}

int main() {
    torch::manual_seed(42);

    int input_dim = 20, hidden_dim = 64, num_classes = 5, num_samples = 200;
    auto data = torch::randn({num_samples, input_dim});
    auto labels = torch::randint(0, num_classes, {num_samples});

    auto model = std::make_shared<SimpleNet>(input_dim, hidden_dim, num_classes);

    // Initial training at higher LR to learn good representations
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

    float baseline = evaluate(*model, data, labels);
    int64_t total_params = 0;
    for (const auto& p : model->parameters()) total_params += p.numel();
    std::cout << "\nBaseline - Accuracy: " << baseline
              << "% | Parameters: " << total_params << std::endl;

    // Sparsity schedule: progressively prune 30% → 50% → 70% → 90%
    // Each step builds on the previous — the model is not reset between rounds
    std::vector<float> schedule = {0.3f, 0.5f, 0.7f, 0.9f};
    iterative_pruning_pipeline(*model, data, labels, schedule, /*finetune_epochs=*/20);

    return 0;
}
