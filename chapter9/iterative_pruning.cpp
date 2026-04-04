#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>

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

class MagnitudePruner {
private:
    torch::nn::Module& model;
    float sparsity_target;
    std::vector<torch::Tensor> masks; // binary masks to freeze pruned weights

public:
    MagnitudePruner(torch::nn::Module& model_, float sparsity_)
        : model(model_), sparsity_target(sparsity_) {}

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

    // Compute and store masks, then apply them
    void apply_pruning() {
        float threshold = compute_threshold();
        masks.clear();
        for (auto& param : model.parameters()) {
            auto mask = (param.data().abs() >= threshold).to(torch::kFloat);
            masks.push_back(mask);
            param.data().mul_(mask);
        }
    }

    // Re-apply stored masks (used after optimizer step to keep pruned weights at zero)
    void reapply_masks() {
        size_t i = 0;
        for (auto& param : model.parameters()) {
            param.data().mul_(masks[i]);
            ++i;
        }
    }

    float measure_sparsity() {
        int64_t total = 0, zeros = 0;
        for (const auto& param : model.parameters()) {
            total += param.numel();
            zeros += (param.data() == 0).sum().item<int64_t>();
        }
        return static_cast<float>(zeros) / total;
    }
};

float evaluate(SimpleNet& model, torch::Tensor data, torch::Tensor labels) {
    torch::NoGradGuard no_grad;
    auto preds = model.forward(data).argmax(1);
    return preds.eq(labels).sum().item<float>() / labels.size(0) * 100;
}

// Iterative pruning: prune → finetune (with mask) → evaluate, repeat at higher sparsity
void iterative_pruning_pipeline(
    SimpleNet& model,
    torch::Tensor data,
    torch::Tensor labels,
    const std::vector<float>& sparsity_schedule,
    int finetune_epochs
) {
    for (float target_sparsity : sparsity_schedule) {
        std::cout << "\n--- Pruning to " << target_sparsity * 100 << "% sparsity ---" << std::endl;

        // Step 1: Prune
        MagnitudePruner pruner(model, target_sparsity);
        pruner.apply_pruning();
        float achieved = pruner.measure_sparsity();
        std::cout << "Achieved sparsity: " << achieved * 100 << "%" << std::endl;

        float acc_before = evaluate(model, data, labels);
        std::cout << "Accuracy after pruning (before finetune): " << acc_before << "%" << std::endl;

        // Step 2: Finetune with mask re-application
        auto optimizer = torch::optim::Adam(model.parameters(),
            torch::optim::AdamOptions(1e-4));

        for (int epoch = 0; epoch < finetune_epochs; ++epoch) {
            optimizer.zero_grad();
            auto loss = torch::nn::functional::cross_entropy(model.forward(data), labels);
            loss.backward();
            optimizer.step();

            // Re-apply mask so pruned weights stay zero
            pruner.reapply_masks();

            if (epoch % 5 == 0 || epoch == finetune_epochs - 1)
                std::cout << "  Finetune epoch " << epoch
                          << ", Loss: " << loss.item<float>() << std::endl;
        }

        // Step 3: Evaluate after finetuning
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

    // Initial training
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

    // Run iterative prune-finetune pipeline with increasing sparsity
    std::vector<float> schedule = {0.3f, 0.5f, 0.7f, 0.9f};
    iterative_pruning_pipeline(*model, data, labels, schedule, /*finetune_epochs=*/20);

    return 0;
}
