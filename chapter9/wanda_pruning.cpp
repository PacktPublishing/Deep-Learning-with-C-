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

    // Forward that also records intermediate activations
    torch::Tensor forward(torch::Tensor x, std::vector<torch::Tensor>* act_norms = nullptr) {
        x = fc1->forward(x);
        if (act_norms) act_norms->push_back(x.abs().mean(0)); // per-output-neuron mean
        x = torch::relu(x);

        x = fc2->forward(x);
        if (act_norms) act_norms->push_back(x.abs().mean(0));
        x = torch::relu(x);

        x = fc3->forward(x);
        if (act_norms) act_norms->push_back(x.abs().mean(0));
        return x;
    }
};

// Wanda: prune by |weight| * |activation| importance (per row of each weight matrix)
void wanda_prune(SimpleNet& model, torch::Tensor calibration_data, float sparsity) {
    torch::NoGradGuard no_grad;

    // Step 1: Collect activation norms from calibration data
    std::vector<torch::Tensor> act_norms;
    model.forward(calibration_data, &act_norms);

    // Step 2: For each linear layer, compute Wanda importance and prune per-row
    std::vector<torch::nn::Linear> layers = {model.fc1, model.fc2, model.fc3};

    for (size_t l = 0; l < layers.size(); ++l) {
        auto& weight = layers[l]->weight; // [out_features, in_features]
        // act_norms[l] has shape [out_features] — activation magnitude after this layer
        // For Wanda, input activation norm matters: use previous layer's output or raw input
        // We approximate input activation norm per column of weight
        torch::Tensor input_act_norm;
        if (l == 0) {
            // Input activation norm: mean absolute value per feature across calibration batch
            input_act_norm = calibration_data.abs().mean(0); // [in_features]
        } else {
            // Use previous layer's activation norm as proxy for this layer's input
            input_act_norm = act_norms[l - 1]; // [in_features of this layer]
        }

        // importance[i,j] = |W[i,j]| * input_act_norm[j]
        auto importance = weight.data().abs() * input_act_norm.unsqueeze(0);

        // Prune per row: zero out lowest-importance weights in each output row
        int64_t n_cols = importance.size(1);
        int64_t prune_count = static_cast<int64_t>(n_cols * sparsity);

        for (int64_t row = 0; row < importance.size(0); ++row) {
            auto [vals, indices] = importance[row].topk(prune_count, /*dim=*/-1, /*largest=*/false);
            weight.data()[row].index_fill_(0, indices, 0.0f);
        }
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

    // Save original weights for re-loading before each sparsity level
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

    // Use a subset as calibration data (Wanda typically needs only a small calibration set)
    auto calibration_data = data.slice(0, 0, 32);

    std::vector<float> sparsity_levels = {0.3f, 0.5f, 0.7f, 0.9f};

    for (float target : sparsity_levels) {
        // Restore original weights
        size_t idx = 0;
        for (auto& p : model->parameters())
            p.data().copy_(original_weights[idx++]);

        wanda_prune(*model, calibration_data, target);

        float actual_sparsity = measure_sparsity(*model);

        torch::NoGradGuard no_grad;
        auto preds = model->forward(data).argmax(1);
        float acc = preds.eq(labels).sum().item<float>() / num_samples * 100;

        std::cout << "\nWanda sparsity target: " << target * 100 << "%"
                  << " | Actual: " << actual_sparsity * 100 << "%"
                  << " | Accuracy: " << acc << "%" << std::endl;
    }

    return 0;
}
