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

// First-order Taylor approximation for importance
// Importance = |gradient × weight| approximates loss change when zeroing a weight
torch::Tensor compute_taylor_importance(
    const torch::Tensor& weights,
    const torch::Tensor& gradients
) {
    return (gradients * weights).abs();
}

// Apply gradient-based pruning
void prune_by_gradient(
    SimpleNet& model,
    torch::Tensor input,
    torch::Tensor target,
    float sparsity
) {
    // Step 1: Compute gradients
    model.zero_grad();
    auto output = model.forward(input);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    loss.backward();

    // Step 2: Collect all importance scores with parameter/index tracking
    struct WeightEntry {
        float importance;
        size_t param_idx;
        int64_t flat_idx;
    };
    std::vector<WeightEntry> entries;
    std::vector<torch::Tensor> params;

    size_t pidx = 0;
    for (auto& param : model.parameters()) {
        if (param.grad().defined()) {
            auto imp = compute_taylor_importance(param.data(), param.grad()).flatten().cpu();
            auto acc = imp.accessor<float, 1>();
            for (int64_t i = 0; i < acc.size(0); ++i) {
                entries.push_back({acc[i], pidx, i});
            }
            params.push_back(param);
            ++pidx;
        }
    }

    // Step 3: Sort by importance (ascending) and prune lowest
    std::sort(entries.begin(), entries.end(),
              [](const WeightEntry& a, const WeightEntry& b) {
                  return a.importance < b.importance;
              });

    size_t prune_count = static_cast<size_t>(entries.size() * sparsity);

    // Build per-parameter masks initialized to ones
    std::vector<torch::Tensor> masks;
    for (auto& p : params) {
        masks.push_back(torch::ones_like(p.data()).flatten());
    }

    for (size_t i = 0; i < prune_count; ++i) {
        masks[entries[i].param_idx][entries[i].flat_idx] = 0.0f;
    }

    // Apply masks
    for (size_t i = 0; i < params.size(); ++i) {
        params[i].data().mul_(masks[i].reshape(params[i].sizes()));
    }
}

float measure_sparsity(SimpleNet& model) {
    int64_t total = 0, zeros = 0;
    for (const auto& p : model.parameters()) {
        total += p.numel();
        zeros += (p.data() == 0).sum().item<int64_t>();
    }
    return static_cast<float>(zeros) / total;
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

    // Apply gradient-based pruning at increasing sparsity levels
    std::vector<float> sparsity_levels = {0.3f, 0.5f, 0.7f, 0.9f};

    for (float target : sparsity_levels) {
        prune_by_gradient(*model, data, labels, target);

        float actual_sparsity = measure_sparsity(*model);

        torch::NoGradGuard no_grad;
        auto preds = model->forward(data).argmax(1);
        float acc = preds.eq(labels).sum().item<float>() / num_samples * 100;

        std::cout << "\nSparsity target: " << target * 100 << "%"
                  << " | Actual: " << actual_sparsity * 100 << "%"
                  << " | Accuracy: " << acc << "%" << std::endl;
    }

    return 0;
}
