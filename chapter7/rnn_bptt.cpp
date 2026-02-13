#include <vector>
#include <cmath>
#include <iostream>
#include <random>

class RNN {
private:
    int input_size, hidden_size, output_size, seq_length;
    std::vector<std::vector<double>> Wxh, Whh, Why;
    std::vector<double> bh, by;
    
public:
    RNN(int input_sz, int hidden_sz, int output_sz, int seq_len) 
        : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz), seq_length(seq_len) {
        
        // Initialize weights randomly
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 0.1);
        
        Wxh.assign(hidden_size, std::vector<double>(input_size));
        Whh.assign(hidden_size, std::vector<double>(hidden_size));
        Why.assign(output_size, std::vector<double>(hidden_size));
        
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size; ++j) Wxh[i][j] = dist(gen);
            for (int j = 0; j < hidden_size; ++j) Whh[i][j] = dist(gen);
        }
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) Why[i][j] = dist(gen);
        }
        
        bh.assign(hidden_size, 0.0);
        by.assign(output_size, 0.0);
    }
    
    double tanh_activation(double x) { return std::tanh(x); }
    double tanh_derivative(double x) { return 1.0 - x * x; }
    
    void forward_backward(const std::vector<std::vector<double>>& inputs,
                         const std::vector<std::vector<double>>& targets,
                         double learning_rate) {
        
        // Forward pass storage
        std::vector<std::vector<double>> h(seq_length + 1, std::vector<double>(hidden_size, 0.0));
        std::vector<std::vector<double>> y(seq_length, std::vector<double>(output_size));
        std::vector<std::vector<double>> p(seq_length, std::vector<double>(output_size));
        
        // Forward pass
        for (int t = 0; t < seq_length; t++) {
            // Hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            for (int i = 0; i < hidden_size; i++) {
                double sum = bh[i];
                for (int j = 0; j < input_size; j++) {
                    sum += Wxh[i][j] * inputs[t][j];
                }
                for (int j = 0; j < hidden_size; j++) {
                    sum += Whh[i][j] * h[t][j];
                }
                h[t + 1][i] = tanh_activation(sum);
            }
            
            // Output: y_t = W_hy * h_t + b_y, p_t = σ(y_t)
            for (int i = 0; i < output_size; i++) {
                double sum = by[i];
                for (int j = 0; j < hidden_size; j++) {
                    sum += Why[i][j] * h[t + 1][j];
                }
                y[t][i] = sum;
                p[t][i] = 1.0 / (1.0 + std::exp(-y[t][i])); // σ(y_t)
            }
        }
        
        // Backward pass - gradients
        std::vector<std::vector<double>> dWxh(hidden_size, std::vector<double>(input_size, 0.0));
        std::vector<std::vector<double>> dWhh(hidden_size, std::vector<double>(hidden_size, 0.0));
        std::vector<std::vector<double>> dWhy(output_size, std::vector<double>(hidden_size, 0.0));
        std::vector<double> dbh(hidden_size, 0.0);
        std::vector<double> dby(output_size, 0.0);
        std::vector<double> dh_next(hidden_size, 0.0);
        
        // Backpropagation through time
        for (int t = seq_length - 1; t >= 0; t--) {
            // Output layer gradients: ∂L/∂y_t = p_t - target_t
            std::vector<double> dy(output_size);
            for (int i = 0; i < output_size; i++) {
                dy[i] = p[t][i] - targets[t][i];  // ∂L/∂y_t
                dby[i] += dy[i];                  // ∂L/∂b_y = ∂L/∂y_t
                for (int j = 0; j < hidden_size; j++) {
                    dWhy[i][j] += dy[i] * h[t + 1][j];  // ∂L/∂W_hy = ∂L/∂y_t * h_t
                }
            }
            
            // Hidden layer gradients: ∂L/∂h_t = W_hy^T * ∂L/∂y_t + ∂L/∂h_{t+1}
            std::vector<double> dh(hidden_size, 0.0);
            for (int i = 0; i < hidden_size; i++) {
                for (int j = 0; j < output_size; j++) {
                    dh[i] += Why[j][i] * dy[j];  // W_hy^T * ∂L/∂y_t
                }
                dh[i] += dh_next[i];  // + ∂L/∂h_{t+1} (from future timestep)
            }
            
            // Raw hidden state gradients: ∂L/∂z_t = ∂L/∂h_t * tanh'(z_t)
            std::vector<double> dh_raw(hidden_size);
            for (int i = 0; i < hidden_size; i++) {
                dh_raw[i] = dh[i] * tanh_derivative(h[t + 1][i]);  // ∂L/∂z_t
                dbh[i] += dh_raw[i];  // ∂L/∂b_h = ∂L/∂z_t
            }
            
            // Weight gradients: ∂L/∂W = ∂L/∂z_t * input^T
            for (int i = 0; i < hidden_size; i++) {
                for (int j = 0; j < input_size; j++) {
                    dWxh[i][j] += dh_raw[i] * inputs[t][j];  // ∂L/∂W_xh = ∂L/∂z_t * x_t^T
                }
                for (int j = 0; j < hidden_size; j++) {
                    dWhh[i][j] += dh_raw[i] * h[t][j];  // ∂L/∂W_hh = ∂L/∂z_t * h_{t-1}^T
                }
            }
            
            // Prepare gradients for next timestep: ∂L/∂h_{t-1} = W_hh^T * ∂L/∂z_t
            for (int i = 0; i < hidden_size; i++) {
                dh_next[i] = 0.0;
                for (int j = 0; j < hidden_size; j++) {
                    dh_next[i] += Whh[j][i] * dh_raw[j];  // W_hh^T * ∂L/∂z_t
                }
            }
        }
        
        // Update weights: θ = θ - α * ∂L/∂θ
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < input_size; j++) {
                Wxh[i][j] -= learning_rate * dWxh[i][j];  // W_xh = W_xh - α * ∂L/∂W_xh
            }
            for (int j = 0; j < hidden_size; j++) {
                Whh[i][j] -= learning_rate * dWhh[i][j];  // W_hh = W_hh - α * ∂L/∂W_hh
            }
            bh[i] -= learning_rate * dbh[i];  // b_h = b_h - α * ∂L/∂b_h
        }
        
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                Why[i][j] -= learning_rate * dWhy[i][j];  // W_hy = W_hy - α * ∂L/∂W_hy
            }
            by[i] -= learning_rate * dby[i];  // b_y = b_y - α * ∂L/∂b_y
        }
    }
    
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& inputs) {
        std::vector<std::vector<double>> outputs(seq_length, std::vector<double>(output_size));
        std::vector<double> h(hidden_size, 0.0);
        
        for (int t = 0; t < seq_length; ++t) {
            // Hidden state
            std::vector<double> h_new(hidden_size);
            for (int i = 0; i < hidden_size; ++i) {
                double sum = bh[i];
                for (int j = 0; j < input_size; ++j) {
                    sum += Wxh[i][j] * inputs[t][j];
                }
                for (int j = 0; j < hidden_size; ++j) {
                    sum += Whh[i][j] * h[j];
                }
                h_new[i] = tanh_activation(sum);
            }
            h = h_new;
            
            // Output
            for (int i = 0; i < output_size; ++i) {
                double sum = by[i];
                for (int j = 0; j < hidden_size; ++j) {
                    sum += Why[i][j] * h[j];
                }
                outputs[t][i] = 1.0 / (1.0 + std::exp(-sum));
            }
        }
        return outputs;
    }
};

int main() {
    // Simple XOR-like sequence problem
    RNN rnn(2, 10, 1, 3);
    
    // Training data: sequences of 3 timesteps
    std::vector<std::vector<std::vector<double>>> train_inputs = {
        {{1, 0}, {0, 1}, {1, 1}},
        {{0, 1}, {1, 0}, {0, 0}},
        {{1, 1}, {0, 0}, {1, 0}}
    };
    
    std::vector<std::vector<std::vector<double>>> train_targets = {
        {{1}, {1}, {0}},
        {{1}, {1}, {0}},
        {{0}, {0}, {1}}
    };
    
    // Training
    std::cout << "Training RNN with BPTT..." << std::endl;
    for (int epoch = 0; epoch < 1000; ++epoch) {
        for (size_t i = 0; i < train_inputs.size(); ++i) {
            rnn.forward_backward(train_inputs[i], train_targets[i], 0.1);
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " completed" << std::endl;
        }
    }
    
    // Test
    std::cout << "\nTesting:" << std::endl;
    for (size_t i = 0; i < train_inputs.size(); ++i) {
        auto outputs = rnn.predict(train_inputs[i]);
        std::cout << "Input sequence " << i << ":" << std::endl;
        for (int t = 0; t < 3; ++t) {
            std::cout << "  t=" << t << ": output=" << outputs[t][0] 
                     << ", target=" << train_targets[i][t][0] << std::endl;
        }
    }
    
    return 0;
}