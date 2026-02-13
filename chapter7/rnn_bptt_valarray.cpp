#include <iostream>
#include <vector>
#include "matrix_ops.h"

class RNN_Valarray {
private:
    int input_size, hidden_size, output_size, seq_length;
    Matrix Wxh, Whh, Why, bh, by;
    
public:
    RNN_Valarray(int input_sz, int hidden_sz, int output_sz, int seq_len) 
        : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz), seq_length(seq_len),
          Wxh(hidden_sz, input_sz), Whh(hidden_sz, hidden_sz), Why(output_sz, hidden_sz),
          bh(hidden_sz, 1), by(output_sz, 1) {
        
        // Initialize weights randomly
        Wxh.randomize();
        Whh.randomize();
        Why.randomize();
        bh.zeros();
        by.zeros();
    }
    
    void forward_backward(const std::vector<Matrix>& inputs,
                         const std::vector<Matrix>& targets,
                         double learning_rate) {
        
        // Forward pass storage
        std::vector<Matrix> h(seq_length + 1, Matrix(hidden_size, 1, 0.0));
        std::vector<Matrix> y(seq_length, Matrix(output_size, 1));
        std::vector<Matrix> p(seq_length, Matrix(output_size, 1));
        std::vector<Matrix> z(seq_length, Matrix(hidden_size, 1));
        
        // Forward pass
        for (int t = 0; t < seq_length; t++) {
            z[t] = Wxh * inputs[t] + Whh * h[t] + bh;
            h[t + 1] = z[t].tanh();
            y[t] = Why * h[t + 1] + by;
            p[t] = y[t].sigmoid();
        }
        
        // Initialize gradients
        Matrix dWxh(hidden_size, input_size, 0.0);
        Matrix dWhh(hidden_size, hidden_size, 0.0);
        Matrix dWhy(output_size, hidden_size, 0.0);
        Matrix dbh(hidden_size, 1, 0.0);
        Matrix dby(output_size, 1, 0.0);
        Matrix dh_next(hidden_size, 1, 0.0);
        
        // Backpropagation through time
        for (int t = seq_length - 1; t >= 0; t--) {
            // Output gradients
            Matrix dy = p[t] - targets[t];
            dby = dby + dy;
            dWhy = dWhy + dy * h[t + 1].transpose();
            
            // Hidden gradients
            Matrix dh = Why.transpose() * dy + dh_next;
            Matrix dh_raw = dh.hadamard(z[t].tanh_derivative());
            dbh = dbh + dh_raw;
            
            // Weight gradients
            dWxh = dWxh + dh_raw * inputs[t].transpose();
            dWhh = dWhh + dh_raw * h[t].transpose();
            
            // Gradient for next timestep
            dh_next = Whh.transpose() * dh_raw;
        }
        
        // Parameter updates
        Wxh = Wxh - dWxh * learning_rate;
        Whh = Whh - dWhh * learning_rate;
        Why = Why - dWhy * learning_rate;
        bh = bh - dbh * learning_rate;
        by = by - dby * learning_rate;
    }
    
    std::vector<Matrix> predict(const std::vector<Matrix>& inputs) {
        std::vector<Matrix> outputs;
        Matrix h(hidden_size, 1, 0.0);
        
        for (int t = 0; t < seq_length; ++t) {
            Matrix z = Wxh * inputs[t] + Whh * h + bh;
            h = z.tanh();
            Matrix y = Why * h + by;
            outputs.push_back(y.sigmoid());
        }
        return outputs;
    }
};

int main() {
    RNN_Valarray rnn(2, 10, 1, 3);
    
    // Same training data as basic RNN
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
    
    // Convert to Matrix format
    std::vector<std::vector<Matrix>> train_inputs;
    std::vector<std::vector<Matrix>> train_targets;
    
    for (size_t i = 0; i < basic_inputs.size(); ++i) {
        std::vector<Matrix> input_seq, target_seq;
        
        for (int t = 0; t < 3; ++t) {
            Matrix input(2, 1);
            input(0, 0) = basic_inputs[i][t][0];
            input(1, 0) = basic_inputs[i][t][1];
            input_seq.push_back(input);
            
            Matrix target(1, 1);
            target(0, 0) = basic_targets[i][t][0];
            target_seq.push_back(target);
        }
        
        train_inputs.push_back(input_seq);
        train_targets.push_back(target_seq);
    }
    
    // Training
    std::cout << "Training Valarray RNN with BPTT..." << std::endl;
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
            std::cout << "  t=" << t << ": output=" << outputs[t](0, 0) 
                     << ", target=" << train_targets[i][t](0, 0) << std::endl;
        }
    }
    
    return 0;
}