#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <functional>
#include <random>
#include <iostream>

// Activation functions
class ActivationFunctions {
public:
    static Eigen::MatrixXf tanh(const Eigen::MatrixXf& x) {
        return x.array().tanh();
    }
    
    static Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& x) {
        return 1.0f / (1.0f + (-x.array()).exp());
    }
    
    static Eigen::MatrixXf relu(const Eigen::MatrixXf& x) {
        return x.cwiseMax(0.0f);
    }
    
    // Derivatives for backpropagation
    static Eigen::MatrixXf tanh_derivative(const Eigen::MatrixXf& x) {
        Eigen::MatrixXf tanh_x = tanh(x);
        return (1.0f - tanh_x.cwiseProduct(tanh_x).array()).matrix();
    }
    
    static Eigen::MatrixXf sigmoid_derivative(const Eigen::MatrixXf& x) {
        Eigen::MatrixXf sig_x = sigmoid(x);
        return (sig_x.array() * (1.0f - sig_x.array())).matrix();
    }
};

int main() {
    Eigen::MatrixXf test(3, 3);
    test << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;
    
    std::cout << "Original matrix:\n" << test << std::endl;
    std::cout << "\nTanh:\n" << ActivationFunctions::tanh(test) << std::endl;
    std::cout << "\nSigmoid:\n" << ActivationFunctions::sigmoid(test) << std::endl;
    std::cout << "\nReLU:\n" << ActivationFunctions::relu(test) << std::endl;
    
    return 0;
}