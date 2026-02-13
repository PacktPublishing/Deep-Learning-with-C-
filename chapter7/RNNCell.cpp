#include <Eigen/Dense>
#include <functional>
#include <random>
#include <cmath>

class RNNCell {
private:
    int input_size;
    int hidden_size;
    int output_size;
    
    // Weight matrices
    Eigen::MatrixXf W_xh;  // Input to hidden
    Eigen::MatrixXf W_hh;  // Hidden to hidden (recurrent)
    Eigen::MatrixXf W_hy;  // Hidden to output
    
    // Bias vectors
    Eigen::VectorXf b_h;   // Hidden bias
    Eigen::VectorXf b_y;   // Output bias
    
    // Activation functions
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> hidden_activation;
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> output_activation;
    
public:
    RNNCell(int input_sz, int hidden_sz, int output_sz) 
        : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz) {
        
        // Initialize weights with Xavier initialization
        initializeWeights();
        
        // Set default activation functions
        hidden_activation = ActivationFunctions::tanh;
        output_activation = ActivationFunctions::sigmoid;
    }
    
private:
    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        float xavier_xh = std::sqrt(6.0f / (input_size + hidden_size));
        float xavier_hh = std::sqrt(6.0f / (hidden_size + hidden_size));
        float xavier_hy = std::sqrt(6.0f / (hidden_size + output_size));
        
        std::uniform_real_distribution<float> dist_xh(-xavier_xh, xavier_xh);
        std::uniform_real_distribution<float> dist_hh(-xavier_hh, xavier_hh);
        std::uniform_real_distribution<float> dist_hy(-xavier_hy, xavier_hy);
        
        W_xh = Eigen::MatrixXf::Zero(hidden_size, input_size).unaryExpr([&](float) { return dist_xh(gen); });
        W_hh = Eigen::MatrixXf::Zero(hidden_size, hidden_size).unaryExpr([&](float) { return dist_hh(gen); });
        W_hy = Eigen::MatrixXf::Zero(output_size, hidden_size).unaryExpr([&](float) { return dist_hy(gen); });
        
        b_h = Eigen::VectorXf::Zero(hidden_size);
        b_y = Eigen::VectorXf::Zero(output_size);
    }
    
public:
    std::pair<Eigen::VectorXf, Eigen::VectorXf> forward(const Eigen::VectorXf& x, const Eigen::VectorXf& h_prev) {
        Eigen::VectorXf z = W_xh * x + W_hh * h_prev + b_h;
        Eigen::VectorXf h = z.unaryExpr([](float val) { return std::tanh(val); });
        Eigen::VectorXf y = W_hy * h + b_y;
        Eigen::VectorXf output = y.unaryExpr([](float val) { return 1.0f / (1.0f + std::exp(-val)); });
        return {h, output};
    }
};

namespace ActivationFunctions {
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> tanh = [](const Eigen::MatrixXf& x) {
        return x.unaryExpr([](float val) { return std::tanh(val); });
    };
    
    std::function<Eigen::MatrixXf(const Eigen::MatrixXf&)> sigmoid = [](const Eigen::MatrixXf& x) {
        return x.unaryExpr([](float val) { return 1.0f / (1.0f + std::exp(-val)); });
    };
}
