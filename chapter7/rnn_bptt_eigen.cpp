#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>

using namespace Eigen;

class RNN_Eigen {
private:
    int input_size, hidden_size, output_size, seq_length;
    MatrixXd Wxh, Whh, Why;
    VectorXd bh, by;
    
public:
    RNN_Eigen(int input_sz, int hidden_sz, int output_sz, int seq_len) 
        : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz), seq_length(seq_len) {
        
        // Initialize weights randomly
        Wxh = MatrixXd::Random(hidden_sz, input_sz) * 0.1;
        Whh = MatrixXd::Random(hidden_sz, hidden_sz) * 0.1;
        Why = MatrixXd::Random(output_sz, hidden_sz) * 0.1;
        
        bh = VectorXd::Zero(hidden_sz);
        by = VectorXd::Zero(output_sz);
    }
    
    void forward_backward(const std::vector<VectorXd>& inputs,
                         const std::vector<VectorXd>& targets,
                         double learning_rate) {
        
        // Forward pass storage
        std::vector<VectorXd> h(seq_length + 1);
        std::vector<VectorXd> y(seq_length);
        std::vector<VectorXd> p(seq_length);
        std::vector<VectorXd> z(seq_length);
        
        h[0] = VectorXd::Zero(hidden_size);
        
        // Forward pass
        for (int t = 0; t < seq_length; t++) {
            z[t] = Wxh * inputs[t] + Whh * h[t] + bh;
            h[t + 1] = z[t].array().tanh();
            y[t] = Why * h[t + 1] + by;
            p[t] = (1.0 / (1.0 + (-y[t].array()).exp()));
        }
        
        // Initialize gradients
        MatrixXd dWxh = MatrixXd::Zero(hidden_size, input_size);
        MatrixXd dWhh = MatrixXd::Zero(hidden_size, hidden_size);
        MatrixXd dWhy = MatrixXd::Zero(output_size, hidden_size);
        VectorXd dbh = VectorXd::Zero(hidden_size);
        VectorXd dby = VectorXd::Zero(output_size);
        VectorXd dh_next = VectorXd::Zero(hidden_size);
        
        // Backpropagation through time
        for (int t = seq_length - 1; t >= 0; t--) {
            // Output gradients
            VectorXd dy = p[t] - targets[t];
            dby += dy;
            dWhy += dy * h[t + 1].transpose();
            
            // Hidden gradients
            VectorXd dh = Why.transpose() * dy + dh_next;
            VectorXd tanh_z = z[t].array().tanh();
            VectorXd dh_raw = dh.cwiseProduct(1.0 - tanh_z.cwiseProduct(tanh_z));
            dbh += dh_raw;
            
            // Weight gradients
            dWxh += dh_raw * inputs[t].transpose();
            dWhh += dh_raw * h[t].transpose();
            
            // Gradient for next timestep
            dh_next = Whh.transpose() * dh_raw;
        }
        
        // Parameter updates
        Wxh -= learning_rate * dWxh;
        Whh -= learning_rate * dWhh;
        Why -= learning_rate * dWhy;
        bh -= learning_rate * dbh;
        by -= learning_rate * dby;
    }
    
    std::vector<VectorXd> predict(const std::vector<VectorXd>& inputs) {
        std::vector<VectorXd> outputs;
        VectorXd h = VectorXd::Zero(hidden_size);
        
        for (int t = 0; t < seq_length; ++t) {
            VectorXd z = Wxh * inputs[t] + Whh * h + bh;
            h = z.array().tanh();
            VectorXd y = Why * h + by;
            outputs.push_back((1.0 / (1.0 + (-y.array()).exp())));
        }
        return outputs;
    }
};

int main() {
    RNN_Eigen rnn(2, 10, 1, 3);
    
    std::vector<std::vector<std::vector<double>>> basic_inputs = {
        {{1, 0}, {0, 1}, {1, 1}},
        {{0, 1}, {1, 0}, {0, 0}},
        {{1, 1}, {0, 0}, {1, 0}}
    };
    
    std::vector<std::vector<std::vector<double>>> basic_targets = {
        {{1}, {1}, {0}},
        {{1}, {1}, {0}},
        {{0}, {0}, {1}}
    };
    
    // Convert to Eigen format
    std::vector<std::vector<VectorXd>> train_inputs;
    std::vector<std::vector<VectorXd>> train_targets;
    
    for (size_t i = 0; i < basic_inputs.size(); ++i) {
        std::vector<VectorXd> input_seq, target_seq;
        
        for (int t = 0; t < 3; ++t) {
            VectorXd input(2);
            input << basic_inputs[i][t][0], basic_inputs[i][t][1];
            input_seq.push_back(input);
            
            VectorXd target(1);
            target << basic_targets[i][t][0];
            target_seq.push_back(target);
        }
        
        train_inputs.push_back(input_seq);
        train_targets.push_back(target_seq);
    }
    
    // Training
    std::cout << "Training Eigen RNN with BPTT..." << std::endl;
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
            std::cout << "  t=" << t << ": output=" << outputs[t](0) 
                     << ", target=" << train_targets[i][t](0) << std::endl;
        }
    }
    
    return 0;
}