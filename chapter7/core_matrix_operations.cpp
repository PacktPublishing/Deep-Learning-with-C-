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
        return 1.0f - tanh_x.cwiseProduct(tanh_x);
    }
    
    static Eigen::MatrixXf sigmoid_derivative(const Eigen::MatrixXf& x) {
        Eigen::MatrixXf sig_x = sigmoid(x);
        return sig_x.cwiseProduct(1.0f - sig_x.array());
    }
};