#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>

class Layer {
private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;
    Eigen::MatrixXd activation;
    Eigen::MatrixXd input;
    Eigen::MatrixXd output;

public:
    Layer(int input_size, int output_size) {
        weights = Eigen::MatrixXd::Random(output_size, input_size) * 0.1;
        biases = Eigen::VectorXd::Zero(output_size);
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) {
        input = x;
        activation = (weights * input + biases.replicate(1, input.cols()));
        output = relu(activation);
        return output;
    }

    Eigen::MatrixXd backward(const Eigen::MatrixXd& grad_output, double learning_rate) {
        Eigen::MatrixXd grad_activation = grad_output.array() * relu_derivative(activation).array();
        
        // Calculate gradients
        Eigen::MatrixXd grad_weights = grad_activation * input.transpose();
        Eigen::VectorXd grad_biases = grad_activation.rowwise().sum();
        Eigen::MatrixXd grad_input = weights.transpose() * grad_activation;

        // Update weights and biases
        weights -= learning_rate * grad_weights;
        biases -= learning_rate * grad_biases;

        return grad_input;
    }

    Eigen::MatrixXd relu(const Eigen::MatrixXd& x) {
        return x.array().max(0);
    }

    Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& x) {
        return (x.array() > 0).cast<double>();
    }
};

class MultilayerPerceptron {
private:
    std::vector<Layer> layers;
    double learning_rate;

public:
    MultilayerPerceptron(const std::vector<int>& layer_sizes, double lr = 0.01)
         : learning_rate(lr) {
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i + 1]);
        }
    }

    Eigen::MatrixXd forward(const Eigen::MatrixXd& input) {
        Eigen::MatrixXd current_input = input;
        for (auto& layer : layers) {
            current_input = layer.forward(current_input);
        }
        return current_input;
    }

    void train(const Eigen::MatrixXd& X, const Eigen::MatrixXd& y, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Forward pass
            auto prediction = forward(X);
            
            // Calculate loss (MSE)
            double loss = (prediction - y).array().square().mean();
            
            if (epoch % 100 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
            }

            // Backpropagation
            Eigen::MatrixXd grad = 2.0 * (prediction - y) / y.cols(); // MSE derivative
            
            // Backward pass through all layers in reverse
            for (int i = layers.size() - 1; i >= 0; --i) {
                grad = layers[i].backward(grad, learning_rate);
            }
        }
    }

    // Method to make predictions after training
    Eigen::MatrixXd predict(const Eigen::MatrixXd& X) {
        return forward(X);
    }
};

int main() {
    // Example usage for XOR problem
    std::vector<int> layer_sizes = {2, 4, 3, 1}; // 2 input, 2 hidden layers, 1 output
    MultilayerPerceptron mlp(layer_sizes, 0.1); // Learning rate = 0.1

    // Create XOR training data
    Eigen::MatrixXd X(2, 4);
    X << 0, 0, 1, 1,
         0, 1, 0, 1;
    Eigen::MatrixXd y(1, 4);
    y << 0, 1, 1, 0;

    // Train the network
    std::cout << "Training the network..." << std::endl;
    mlp.train(X, y, 1000);

    // Test the network
    std::cout << "\nTesting the network..." << std::endl;
    Eigen::MatrixXd predictions = mlp.predict(X);
    std::cout << "Predictions:" << std::endl;
    std::cout << predictions << std::endl;

    return 0;
}
