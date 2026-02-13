#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <Eigen/Dense>

class EigenLSTMCell {
private:
    int input_size, hidden_size;
    
    // Combined weight matrices for efficient computation
    Eigen::MatrixXf W_combined;  // [4*hidden_size, input_size]
    Eigen::MatrixXf U_combined;  // [4*hidden_size, hidden_size]
    Eigen::VectorXf b_combined;  // [4*hidden_size]
    
public:
    EigenLSTMCell(int input_sz, int hidden_sz) 
        : input_size(input_sz), hidden_size(hidden_sz) {
        
        // Initialize combined matrices for efficient computation [[11](https://www.skytowner.com/explore/comprehensive_guide_on_lstm)]
        W_combined = Eigen::MatrixXf::Random(4 * hidden_size, input_size) * 
                    std::sqrt(2.0f / (input_size + hidden_size));
        U_combined = Eigen::MatrixXf::Random(4 * hidden_size, hidden_size) * 
                    std::sqrt(2.0f / (hidden_size + hidden_size));
        
        b_combined = Eigen::VectorXf::Zero(4 * hidden_size);
        // Set forget gate bias to 1.0 for better gradient flow
        b_combined.segment(0, hidden_size).setOnes();
    }
    
    std::pair<Eigen::VectorXf, Eigen::VectorXf> forward(
        const Eigen::VectorXf& input,
        const Eigen::VectorXf& prev_hidden,
        const Eigen::VectorXf& prev_cell) {
        
        // Efficient matrix operations: compute all gates simultaneously
        Eigen::VectorXf gates = W_combined * input + U_combined * prev_hidden + b_combined;
        
        // Extract individual gates using efficient block operations
        Eigen::VectorXf forget_gate = gates.segment(0, hidden_size);
        Eigen::VectorXf input_gate = gates.segment(hidden_size, hidden_size);
        Eigen::VectorXf candidate_gate = gates.segment(2 * hidden_size, hidden_size);
        Eigen::VectorXf output_gate = gates.segment(3 * hidden_size, hidden_size);
        
        // Apply activation functions using vectorized operations
        forget_gate = forget_gate.unaryExpr([](float x) { 
            return 1.0f / (1.0f + std::exp(-std::max(-50.0f, std::min(50.0f, x)))); 
        });
        
        input_gate = input_gate.unaryExpr([](float x) { 
            return 1.0f / (1.0f + std::exp(-std::max(-50.0f, std::min(50.0f, x)))); 
        });
        
        candidate_gate = candidate_gate.unaryExpr([](float x) { 
            return std::tanh(std::max(-10.0f, std::min(10.0f, x))); 
        });
        
        output_gate = output_gate.unaryExpr([](float x) { 
            return 1.0f / (1.0f + std::exp(-std::max(-50.0f, std::min(50.0f, x)))); 
        });
        
        // Update cell state using element-wise operations
        Eigen::VectorXf new_cell = forget_gate.cwiseProduct(prev_cell) + 
                                  input_gate.cwiseProduct(candidate_gate);
        
        // Update hidden state
        Eigen::VectorXf new_hidden = output_gate.cwiseProduct(
            new_cell.unaryExpr([](float x) { 
                return std::tanh(std::max(-10.0f, std::min(10.0f, x))); 
            })
        );
        
        return std::make_pair(new_hidden, new_cell);
    }
};

class LSTMNetwork {
private:
    EigenLSTMCell lstm_cell;
    Eigen::MatrixXf W_output;
    Eigen::VectorXf b_output;
    int hidden_size;
    
public:
    LSTMNetwork(int input_size, int hidden_sz) 
        : lstm_cell(input_size, hidden_sz), hidden_size(hidden_sz) {
        W_output = Eigen::MatrixXf::Random(1, hidden_size) * 0.1f;
        b_output = Eigen::VectorXf::Zero(1);
    }
    
    float forward(const std::vector<Eigen::VectorXf>& sequence) {
        Eigen::VectorXf hidden = Eigen::VectorXf::Zero(hidden_size);
        Eigen::VectorXf cell = Eigen::VectorXf::Zero(hidden_size);
        
        for (const auto& input : sequence) {
            auto result = lstm_cell.forward(input, hidden, cell);
            hidden = result.first;
            cell = result.second;
        }
        
        return (W_output * hidden + b_output)(0);
    }
};

int main() {
    LSTMNetwork network(1, 20);
    
    // Time series prediction: predict next value in sine wave
    std::vector<std::vector<Eigen::VectorXf>> sequences;
    std::vector<float> targets;
    
    // Generate sine wave training data
    for (int i = 0; i < 50; ++i) {
        std::vector<Eigen::VectorXf> seq;
        for (int t = 0; t < 4; ++t) {
            float val = std::sin((i + t) * 0.1f);
            seq.push_back(Eigen::VectorXf{{val}});
        }
        sequences.push_back(seq);
        targets.push_back(std::sin((i + 4) * 0.1f));
    }
    
    // Training
    for (int epoch = 0; epoch < 10000; ++epoch) {
        float total_loss = 0.0f;
        
        for (size_t i = 0; i < sequences.size(); ++i) {
            float pred = network.forward(sequences[i]);
            float error = pred - targets[i];
            total_loss += error * error;
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", MSE: " << total_loss / sequences.size() << std::endl;
        }
    }
    
    // Test on new sequence
    std::vector<Eigen::VectorXf> test_seq = {
        {Eigen::VectorXf{{std::sin(5.0f * 0.1f)}}},
        {Eigen::VectorXf{{std::sin(5.1f * 0.1f)}}},
        {Eigen::VectorXf{{std::sin(5.2f * 0.1f)}}},
        {Eigen::VectorXf{{std::sin(5.3f * 0.1f)}}}
    };
    
    float prediction = network.forward(test_seq);
    float actual = std::sin(5.4f * 0.1f);
    
    std::cout << "\nTime Series Prediction:" << std::endl;
    std::cout << "Predicted: " << prediction << std::endl;
    std::cout << "Actual: " << actual << std::endl;
    std::cout << "Error: " << std::abs(prediction - actual) << std::endl;
    
    return 0;
}
