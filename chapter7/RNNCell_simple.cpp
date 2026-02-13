#include <vector>
#include <random>
#include <iostream>
#include <cmath>

class RNNCell {
private:
    int input_size, hidden_size, output_size;
    std::vector<std::vector<float>> W_xh, W_hh, W_hy;
    std::vector<float> b_h, b_y;
    
public:
    RNNCell(int input_sz, int hidden_sz, int output_sz) 
        : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz) {
        
        initializeWeights();
    }
    
    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        W_xh.assign(hidden_size, std::vector<float>(input_size));
        W_hh.assign(hidden_size, std::vector<float>(hidden_size));
        W_hy.assign(output_size, std::vector<float>(hidden_size));
        
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size; ++j) W_xh[i][j] = dist(gen);
            for (int j = 0; j < hidden_size; ++j) W_hh[i][j] = dist(gen);
        }
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) W_hy[i][j] = dist(gen);
        }
        
        b_h.assign(hidden_size, 0.0f);
        b_y.assign(output_size, 0.0f);
    }
    
    std::pair<std::vector<float>, std::vector<float>> forward(
        const std::vector<float>& input, 
        const std::vector<float>& prev_hidden) {
        
        std::vector<float> hidden_state(hidden_size);
        
        // h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
        for (int i = 0; i < hidden_size; ++i) {
            float sum = b_h[i];
            for (int j = 0; j < input_size; ++j) {
                sum += W_xh[i][j] * input[j];
            }
            for (int j = 0; j < hidden_size; ++j) {
                sum += W_hh[i][j] * prev_hidden[j];
            }
            hidden_state[i] = std::tanh(sum);
        }
        
        std::vector<float> output(output_size);
        
        // y_t = sigmoid(W_hy * h_t + b_y)
        for (int i = 0; i < output_size; ++i) {
            float sum = b_y[i];
            for (int j = 0; j < hidden_size; ++j) {
                sum += W_hy[i][j] * hidden_state[j];
            }
            output[i] = 1.0f / (1.0f + std::exp(-sum));
        }
        
        return {hidden_state, output};
    }
    
    struct ForwardResult {
        std::vector<std::vector<float>> hidden_states;
        std::vector<std::vector<float>> outputs;
    };
    
    ForwardResult forwardSequence(const std::vector<std::vector<float>>& sequence) {
        ForwardResult result;
        
        std::vector<float> hidden_state(hidden_size, 0.0f);
        result.hidden_states.push_back(hidden_state);
        
        for (const auto& input : sequence) {
            auto [new_hidden, output] = forward(input, hidden_state);
            hidden_state = new_hidden;
            
            result.hidden_states.push_back(hidden_state);
            result.outputs.push_back(output);
        }
        
        return result;
    }
};

int main() {
    RNNCell rnn(2, 3, 1);
    
    std::vector<std::vector<float>> sequence = {
        {0.5f, -0.3f},
        {0.2f, 0.8f},
        {-0.1f, 0.4f}
    };
    
    auto result = rnn.forwardSequence(sequence);
    
    std::cout << "Sequence processing results:\n";
    for (size_t t = 0; t < result.outputs.size(); ++t) {
        std::cout << "Time " << t << ": Input=[";
        for (float x : sequence[t]) std::cout << x << " ";
        std::cout << "] Output=[";
        for (float y : result.outputs[t]) std::cout << y << " ";
        std::cout << "]\n";
    }
    
    return 0;
}