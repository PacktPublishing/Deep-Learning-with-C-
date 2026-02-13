#include <torch/torch.h>
#include <iostream>

// Define the MLP structure
struct MLP : torch::nn::Module {
    torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr};
    
    MLP() {
        // Same architecture as before: 2-4-3-1
        layer1 = register_module("layer1", torch::nn::Linear(2, 4));
        layer2 = register_module("layer2", torch::nn::Linear(4, 3));
        layer3 = register_module("layer3", torch::nn::Linear(3, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(layer1->forward(x));
        x = torch::relu(layer2->forward(x));
        x = layer3->forward(x);
        return x;
    }
};

int main() {
    // Create XOR dataset
    auto X = torch::tensor({
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    });
    
    auto y = torch::tensor({
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    });

    // Create model
    auto model = std::make_shared<MLP>();
    
    // Define optimizer
    torch::optim::SGD optimizer(model->parameters(), 0.1);

    // Training loop
    for (size_t epoch = 0; epoch < 1000; ++epoch) {
        // Forward pass
        auto output = model->forward(X);
        auto loss = torch::mse_loss(output, y);

        // Backward pass and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " 
                      << loss.item<float>() << std::endl;
        }
    }

    // Test the model
    model->eval();
    torch::NoGradGuard no_grad;
    auto predictions = model->forward(X);
    std::cout << "\nFinal Predictions:\n" << predictions << std::endl;

    return 0;
}
