#include <torch/torch.h>
#include <vector>

class RNN_LibTorch : public torch::nn::Module {
private:
    int input_size, hidden_size, output_size, seq_length;
    torch::Tensor Wxh, Whh, Why, bh, by;
    
public:
    RNN_LibTorch(int input_sz, int hidden_sz, int output_sz, int seq_len)
        : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz), seq_length(seq_len) {
        
        // Initialize parameters: W ~ N(0, 0.01), b = 0
        Wxh = register_parameter("Wxh", torch::randn({hidden_size, input_size}) * 0.01);
        Whh = register_parameter("Whh", torch::randn({hidden_size, hidden_size}) * 0.01);
        Why = register_parameter("Why", torch::randn({output_size, hidden_size}) * 0.01);
        bh = register_parameter("bh", torch::zeros({hidden_size, 1}));
        by = register_parameter("by", torch::zeros({output_size, 1}));
    }
    
    std::pair<torch::Tensor, std::vector<torch::Tensor>> forward(const torch::Tensor& inputs) {
        // inputs: [seq_length, batch_size, input_size]
        int batch_size = inputs.size(1);
        
        // Initialize hidden states: h_0 = 0
        torch::Tensor h = torch::zeros({hidden_size, batch_size}, inputs.options());
        std::vector<torch::Tensor> hidden_states;
        std::vector<torch::Tensor> outputs;
        
        hidden_states.push_back(h);
        
        // Forward pass through sequence
        for (int t = 0; t < seq_length; t++) {
            torch::Tensor x_t = inputs[t];  // [batch_size, input_size] -> [input_size, batch_size]
            x_t = x_t.transpose(0, 1);
            
            // h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            torch::Tensor z = torch::mm(Wxh, x_t) + torch::mm(Whh, h) + bh;
            h = torch::tanh(z);
            hidden_states.push_back(h);
            
            // y_t = W_hy * h_t + b_y
            torch::Tensor y = torch::mm(Why, h) + by;
            outputs.push_back(y.transpose(0, 1));  // [batch_size, output_size]
        }
        
        // Stack outputs: [seq_length, batch_size, output_size]
        torch::Tensor output_tensor = torch::stack(outputs, 0);
        return std::make_pair(output_tensor, hidden_states);
    }
    
    torch::Tensor compute_loss(const torch::Tensor& predictions, const torch::Tensor& targets) {
        // Binary cross-entropy loss: L = -Σ[y*log(p) + (1-y)*log(1-p)]
        torch::Tensor probs = torch::sigmoid(predictions);
        return torch::binary_cross_entropy(probs, targets);
    }
    
    void train_step(const torch::Tensor& inputs, const torch::Tensor& targets, 
                   torch::optim::Optimizer& optimizer) {
        
        // Zero gradients
        optimizer.zero_grad();
        
        // Forward pass: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        auto [predictions, hidden_states] = forward(inputs);
        
        // Compute loss: L = BCE(σ(y_t), target_t)
        torch::Tensor loss = compute_loss(predictions, targets);
        
        // Backward pass: automatic differentiation computes ∂L/∂θ
        loss.backward();
        
        // Parameter update: θ = θ - α * ∂L/∂θ
        optimizer.step();
    }
    
    // Manual BPTT implementation for educational purposes
    void manual_bptt(const torch::Tensor& inputs, const torch::Tensor& targets, float learning_rate) {
        
        // Forward pass
        auto [predictions, hidden_states] = forward(inputs);
        int batch_size = inputs.size(1);
        
        // Initialize gradients
        torch::Tensor dWxh = torch::zeros_like(Wxh);
        torch::Tensor dWhh = torch::zeros_like(Whh);
        torch::Tensor dWhy = torch::zeros_like(Why);
        torch::Tensor dbh = torch::zeros_like(bh);
        torch::Tensor dby = torch::zeros_like(by);
        torch::Tensor dh_next = torch::zeros({hidden_size, batch_size}, inputs.options());
        
        // Backpropagation through time
        for (int t = seq_length - 1; t >= 0; t--) {
            torch::Tensor x_t = inputs[t].transpose(0, 1);  // [input_size, batch_size]
            torch::Tensor h_t = hidden_states[t + 1];       // [hidden_size, batch_size]
            torch::Tensor h_prev = hidden_states[t];        // [hidden_size, batch_size]
            torch::Tensor y_t = predictions[t].transpose(0, 1);  // [output_size, batch_size]
            torch::Tensor target_t = targets[t].transpose(0, 1); // [output_size, batch_size]
            
            // Output gradients: ∂L/∂y_t = σ(y_t) - target_t
            torch::Tensor p_t = torch::sigmoid(y_t);
            torch::Tensor dy = p_t - target_t;  // ∂L/∂y_t
            
            // Gradient accumulation
            dby += torch::sum(dy, 1, true);                    // ∂L/∂b_y += Σ(∂L/∂y_t)
            dWhy += torch::mm(dy, h_t.transpose(0, 1));        // ∂L/∂W_hy += ∂L/∂y_t * h_t^T
            
            // Hidden gradients: ∂L/∂h_t = W_hy^T * ∂L/∂y_t + ∂L/∂h_{t+1}
            torch::Tensor dh = torch::mm(Why.transpose(0, 1), dy) + dh_next;
            
            // Raw hidden gradients: ∂L/∂z_t = ∂L/∂h_t ⊙ (1 - h_t²)
            torch::Tensor dh_raw = dh * (1 - h_t * h_t);  // tanh derivative
            
            // Weight gradients
            dbh += torch::sum(dh_raw, 1, true);                // ∂L/∂b_h += Σ(∂L/∂z_t)
            dWxh += torch::mm(dh_raw, x_t.transpose(0, 1));    // ∂L/∂W_xh += ∂L/∂z_t * x_t^T
            dWhh += torch::mm(dh_raw, h_prev.transpose(0, 1)); // ∂L/∂W_hh += ∂L/∂z_t * h_{t-1}^T
            
            // Gradient for next timestep: ∂L/∂h_{t-1} = W_hh^T * ∂L/∂z_t
            dh_next = torch::mm(Whh.transpose(0, 1), dh_raw);
        }
        
        // Manual parameter updates: θ = θ - α * ∂L/∂θ
        {
            torch::NoGradGuard no_grad;
            Wxh -= learning_rate * dWxh;  // W_xh = W_xh - α * ∂L/∂W_xh
            Whh -= learning_rate * dWhh;  // W_hh = W_hh - α * ∂L/∂W_hh
            Why -= learning_rate * dWhy;  // W_hy = W_hy - α * ∂L/∂W_hy
            bh -= learning_rate * dbh;    // b_h = b_h - α * ∂L/∂b_h
            by -= learning_rate * dby;    // b_y = b_y - α * ∂L/∂b_y
        }
    }
};

// Usage example
void train_rnn() {
    // Model configuration
    int input_size = 10, hidden_size = 20, output_size = 5, seq_length = 15, batch_size = 32;
    
    // Create model and move to GPU if available
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    RNN_LibTorch model(input_size, hidden_size, output_size, seq_length);
    model.to(device);
    
    // Create optimizer: Adam with learning rate 0.001
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));
    
    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        // Generate random data: [seq_length, batch_size, input_size]
        torch::Tensor inputs = torch::randn({seq_length, batch_size, input_size}).to(device);
        torch::Tensor targets = torch::randint(0, 2, {seq_length, batch_size, output_size}).to(torch::kFloat).to(device);
        
        // Training step with automatic differentiation
        model.train_step(inputs, targets, optimizer);
        
        // Alternative: Manual BPTT (for educational purposes)
        // model.manual_bptt(inputs, targets, 0.001);
        
        if (epoch % 10 == 0) {
            auto [predictions, _] = model.forward(inputs);
            torch::Tensor loss = model.compute_loss(predictions, targets);
            std::cout << "Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
        }
    }
}