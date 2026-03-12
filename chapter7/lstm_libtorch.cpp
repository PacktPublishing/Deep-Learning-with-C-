#include <torch/torch.h>
#include <iostream>
#include <vector>

class ProductionLSTM : public torch::nn::Module {
private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear output_projection{nullptr};
    torch::nn::Dropout dropout{nullptr};
    int hidden_size;
    int num_layers;
    
public:
    ProductionLSTM(int input_size, int hidden_sz, int num_layers_val, 
                   int output_size, float dropout_rate = 0.2f)
        : hidden_size(hidden_sz), num_layers(num_layers_val) {
        
        // Initialize LSTM
        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(input_size, hidden_size)
                .num_layers(num_layers)
                .dropout(dropout_rate)
                .batch_first(true)
                .bidirectional(false)
        ));
        
        output_projection = register_module("output_projection",
            torch::nn::Linear(hidden_size, output_size));
        
        dropout = register_module("dropout", 
            torch::nn::Dropout(dropout_rate));
    }
    
    torch::Tensor forward(torch::Tensor input) {
        // LSTM forward pass
        auto lstm_result = lstm->forward(input);
        auto lstm_out = std::get<0>(lstm_result);
        
        // Apply dropout for regularization
        lstm_out = dropout->forward(lstm_out);
        
        // Project to output space
        auto output = output_projection->forward(lstm_out);
        
        return output;
    }
    
    float train_step(torch::Tensor input, torch::Tensor target, 
                    torch::optim::Optimizer& optimizer) {
        
        optimizer.zero_grad();
        
        auto predictions = forward(input);
        auto loss = torch::mse_loss(predictions, target);
        loss.backward();
        
        // Gradient clipping to prevent exploding gradients
        torch::nn::utils::clip_grad_norm_(parameters(), 5.0);
        
        optimizer.step();
        
        return loss.item<float>();
    }
};

int main() {
    std::cout << "Testing LibTorch LSTM..." << std::endl;
    
    torch::manual_seed(42);
    
    // Model parameters
    const int input_size = 1;
    const int hidden_size = 64;
    const int num_layers = 2;
    const int output_size = 1;
    const int seq_length = 10;
    const int batch_size = 32;
    const int num_epochs = 100;
    const float learning_rate = 0.001f;
    
    // Create model and move to GPU if available
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    
    auto model = std::make_shared<ProductionLSTM>(input_size, hidden_size, num_layers, output_size);
    model->to(device);
    
    // Optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    
    // Generate sine wave training data
    auto generate_data = [&](int batch_sz, int seq_len) {
        auto x = torch::randn({batch_sz, seq_len, input_size}, device);
        auto y = torch::sin(x * 2 * M_PI) + 0.1 * torch::randn({batch_sz, seq_len, output_size}, device);
        return std::make_pair(x, y);
    };
    
    std::cout << "Training LSTM model..." << std::endl;
    
    // Training loop
    model->train();
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto [input, target] = generate_data(batch_size, seq_length);
        
        float loss = model->train_step(input, target, optimizer);
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }
    
    // Evaluation
    std::cout << "\nEvaluating model..." << std::endl;
    model->eval();
    torch::NoGradGuard no_grad;
    auto [test_input, test_target] = generate_data(1, seq_length);
    auto predictions = model->forward(test_input);
    float test_loss = torch::mse_loss(predictions, test_target).item<float>();
    
    std::cout << "Final test loss: " << test_loss << std::endl;
    std::cout << "LibTorch LSTM test completed successfully!" << std::endl;
    
    return 0;
}
