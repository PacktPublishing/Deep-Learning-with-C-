#include <iostream>
#include <vector>
#include <random>
#include <cmath>

class BasicLSTMCell {
private:
    int input_size;
    int hidden_size;
    
    // Weight matrices for all four gates
    std::vector<std::vector<float>> W_f, W_i, W_c, W_o;  // Input weights
    std::vector<std::vector<float>> U_f, U_i, U_c, U_o;  // Hidden weights
    std::vector<float> b_f, b_i, b_c, b_o;               // Bias vectors
    
    // Activation functions with numerical stability
    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-std::max(-50.0f, std::min(50.0f, x))));
    }
    
    float tanh_activation(float x) {
        return std::tanh(std::max(-10.0f, std::min(10.0f, x)));
    }

public:
    BasicLSTMCell(int input_sz, int hidden_sz) 
        : input_size(input_sz), hidden_size(hidden_sz) {
        
        // Initialize weight matrices with Xavier initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        float xavier_bound = std::sqrt(6.0f / (input_size + hidden_size));
        std::uniform_real_distribution<float> dis(-xavier_bound, xavier_bound);
        
        // Initialize all weight matrices
        W_f.resize(hidden_size, std::vector<float>(input_size));
        W_i.resize(hidden_size, std::vector<float>(input_size));
        W_c.resize(hidden_size, std::vector<float>(input_size));
        W_o.resize(hidden_size, std::vector<float>(input_size));
        
        U_f.resize(hidden_size, std::vector<float>(hidden_size));
        U_i.resize(hidden_size, std::vector<float>(hidden_size));
        U_c.resize(hidden_size, std::vector<float>(hidden_size));
        U_o.resize(hidden_size, std::vector<float>(hidden_size));
        
        // Initialize biases (forget gate bias = 1.0 for better gradient flow)
        b_f.resize(hidden_size, 1.0f);
        b_i.resize(hidden_size, 0.0f);
        b_c.resize(hidden_size, 0.0f);
        b_o.resize(hidden_size, 0.0f);
        
        // Initialize weights with Xavier distribution
        for (auto& matrix : {&W_f, &W_i, &W_c, &W_o}) {
            for (auto& row : *matrix) {
                for (float& val : row) {
                    val = dis(gen);
                }
            }
        }
        
        for (auto& matrix : {&U_f, &U_i, &U_c, &U_o}) {
            for (auto& row : *matrix) {
                for (float& val : row) {
                    val = dis(gen);
                }
            }
        }
    }
    
    // LSTM forward pass implementing all gate equations
    std::pair<std::vector<float>, std::vector<float>> forward(
        const std::vector<float>& input,
        const std::vector<float>& prev_hidden,
        const std::vector<float>& prev_cell) {
        
        std::vector<float> forget_gate(hidden_size);
        std::vector<float> input_gate(hidden_size);
        std::vector<float> candidate_gate(hidden_size);
        std::vector<float> output_gate(hidden_size);
        
        // Compute all gates according to LSTM equations
        for (int h = 0; h < hidden_size; ++h) {
            // Forget gate: f_t = σ(W_f * x_t + U_f * h_{t-1} + b_f)
            float f_sum = b_f[h];
            for (int i = 0; i < input_size; ++i) {
                f_sum += W_f[h][i] * input[i];
            }
            for (int j = 0; j < hidden_size; ++j) {
                f_sum += U_f[h][j] * prev_hidden[j];
            }
            forget_gate[h] = sigmoid(f_sum);
            
            // Input gate: i_t = σ(W_i * x_t + U_i * h_{t-1} + b_i)
            float i_sum = b_i[h];
            for (int i = 0; i < input_size; ++i) {
                i_sum += W_i[h][i] * input[i];
            }
            for (int j = 0; j < hidden_size; ++j) {
                i_sum += U_i[h][j] * prev_hidden[j];
            }
            input_gate[h] = sigmoid(i_sum);
            
            // Candidate gate: C̃_t = tanh(W_c * x_t + U_c * h_{t-1} + b_c)
            float c_sum = b_c[h];
            for (int i = 0; i < input_size; ++i) {
                c_sum += W_c[h][i] * input[i];
            }
            for (int j = 0; j < hidden_size; ++j) {
                c_sum += U_c[h][j] * prev_hidden[j];
            }
            candidate_gate[h] = tanh_activation(c_sum);
            
            // Output gate: o_t = σ(W_o * x_t + U_o * h_{t-1} + b_o)
            float o_sum = b_o[h];
            for (int i = 0; i < input_size; ++i) {
                o_sum += W_o[h][i] * input[i];
            }
            for (int j = 0; j < hidden_size; ++j) {
                o_sum += U_o[h][j] * prev_hidden[j];
            }
            output_gate[h] = sigmoid(o_sum);
        }
        
        // Update cell state: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        std::vector<float> new_cell(hidden_size);
        for (int h = 0; h < hidden_size; ++h) {
            new_cell[h] = forget_gate[h] * prev_cell[h] + 
                         input_gate[h] * candidate_gate[h];
        }
        
        // Update hidden state: h_t = o_t ⊙ tanh(C_t)
        std::vector<float> new_hidden(hidden_size);
        for (int h = 0; h < hidden_size; ++h) {
            new_hidden[h] = output_gate[h] * tanh_activation(new_cell[h]);
        }
        
        return std::make_pair(new_hidden, new_cell);
    }
};

class LSTMNetwork {
private:
    BasicLSTMCell lstm_cell;
    std::vector<std::vector<float>> W_out;
    std::vector<float> b_out;
    int hidden_size, output_size;
    
public:
    LSTMNetwork(int input_sz, int hidden_sz, int output_sz) 
        : lstm_cell(input_sz, hidden_sz), hidden_size(hidden_sz), output_size(output_sz) {
        
        W_out.resize(output_size, std::vector<float>(hidden_size));
        b_out.resize(output_size, 0.0f);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-0.1f, 0.1f);
        
        for (auto& row : W_out) {
            for (float& val : row) {
                val = dis(gen);
            }
        }
    }
    
    std::vector<float> forward(const std::vector<std::vector<float>>& sequence) {
        std::vector<float> hidden(hidden_size, 0.0f);
        std::vector<float> cell(hidden_size, 0.0f);
        
        for (const auto& input : sequence) {
            auto [h, c] = lstm_cell.forward(input, hidden, cell);
            hidden = h;
            cell = c;
        }
        
        std::vector<float> output(output_size);
        for (int i = 0; i < output_size; ++i) {
            output[i] = b_out[i];
            for (int j = 0; j < hidden_size; ++j) {
                output[i] += W_out[i][j] * hidden[j];
            }
        }
        
        return output;
    }
    
    void train(const std::vector<std::vector<std::vector<float>>>& sequences,
               const std::vector<std::vector<float>>& targets, 
               int epochs, float lr) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            
            for (size_t i = 0; i < sequences.size(); ++i) {
                auto output = forward(sequences[i]);
                
                // Compute loss (MSE)
                float loss = 0.0f;
                std::vector<float> grad_output(output_size);
                for (int j = 0; j < output_size; ++j) {
                    float error = output[j] - targets[i][j];
                    loss += error * error;
                    grad_output[j] = 2.0f * error;
                }
                total_loss += loss;
                
                // Simple gradient update for output layer only
                std::vector<float> hidden(hidden_size, 0.0f);
                std::vector<float> cell(hidden_size, 0.0f);
                for (const auto& input : sequences[i]) {
                    auto [h, c] = lstm_cell.forward(input, hidden, cell);
                    hidden = h;
                    cell = c;
                }
                
                for (int j = 0; j < output_size; ++j) {
                    b_out[j] -= lr * grad_output[j];
                    for (int k = 0; k < hidden_size; ++k) {
                        W_out[j][k] -= lr * grad_output[j] * hidden[k];
                    }
                }
            }
            
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << total_loss / sequences.size() << std::endl;
            }
        }
    }
};

int main() {
    LSTMNetwork network(1, 4, 1);
    
    // Training data: sequences and targets
    std::vector<std::vector<std::vector<float>>> sequences = {
        {{0}, {1}, {1}}, {{1}, {0}, {1}}, {{1}, {1}, {0}}
    };
    std::vector<std::vector<float>> targets = {{1}, {0}, {1}};
    
    // Train the network
    network.train(sequences, targets, 1000, 0.01f);
    
    // Test
    auto output = network.forward({{0}, {1}, {1}});
    std::cout << "Final output: " << output[0] << std::endl;
    return 0;
}