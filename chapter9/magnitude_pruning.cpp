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

public:
    MagnitudePruner(torch::nn::Module& model_, float sparsity_)
        : model(model_), sparsity_target(sparsity_) {}

    // Compute global magnitude threshold across all parameters
    float compute_threshold() {
        std::vector<float> all_weights;

        // Step 1: Collect all weight magnitudes
        for (const auto& param : model.parameters()) {
            auto weights = param.data().abs().flatten();
            auto weights_cpu = weights.cpu();
            auto accessor = weights_cpu.accessor<float, 1>();

            for (int64_t i = 0; i < accessor.size(0); ++i) {
                all_weights.push_back(accessor[i]);
            }
        }

        // Step 2: Find threshold for target sparsity
        size_t threshold_idx = static_cast<size_t>(
            all_weights.size() * sparsity_target
        );

        // Step 3: Efficient k-th element selection
        std::nth_element(
            all_weights.begin(),
            all_weights.begin() + threshold_idx,
            all_weights.end()
        );

        return all_weights[threshold_idx];
    }

    // Apply binary mask to parameters below threshold
    void apply_pruning() {
        float threshold = compute_threshold();

        for (auto& param : model.parameters()) {
            // Create mask: 1 for weights above threshold, 0 otherwise
            auto mask = (param.data().abs() >= threshold).to(torch::kFloat);

            // Apply mask in-place
            param.data().mul_(mask);
        }
    }

    // Calculate actual sparsity achieved
    float measure_sparsity() {
        int64_t total_params = 0;
        int64_t zero_params = 0;

        for (const auto& param : model.parameters()) {
            total_params += param.numel();
            zero_params += (param.data() == 0).sum().item<int64_t>();
        }

        return static_cast<float>(zero_params) / total_params;
    }
};

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

    // Evaluate before pruning
    {
        torch::NoGradGuard no_grad;
        auto preds = model->forward(data).argmax(1);
        float acc = preds.eq(labels).sum().item<float>() / num_samples * 100;
        std::cout << "\nBefore pruning - Accuracy: " << acc << "%" << std::endl;
    }

    // Count total parameters
    int64_t total_params = 0;
    for (const auto& p : model->parameters()) total_params += p.numel();
    std::cout << "Total parameters: " << total_params << std::endl;

    // Apply iterative pruning at increasing sparsity levels
    std::vector<float> sparsity_levels = {0.3f, 0.5f, 0.7f, 0.9f};

    for (float target : sparsity_levels) {
        // Reload original weights by re-creating pruner each time on current model
        MagnitudePruner pruner(*model, target);
        pruner.apply_pruning();

        float actual_sparsity = pruner.measure_sparsity();

        torch::NoGradGuard no_grad;
        auto preds = model->forward(data).argmax(1);
        float acc = preds.eq(labels).sum().item<float>() / num_samples * 100;

        std::cout << "\nSparsity target: " << target * 100 << "%"
                  << " | Actual: " << actual_sparsity * 100 << "%"
                  << " | Accuracy: " << acc << "%" << std::endl;
    }

    return 0;
}
