#include <torch/torch.h>

class CNNImpl : public torch::nn::Module {
private:
    torch::nn::Conv2d conv1, conv2, conv3;
    torch::nn::Linear fc1, fc2;
    torch::nn::MaxPool2d pool;
    torch::nn::Dropout dropout;

public:
    CNNImpl(int num_classes = 10) {
        // Convolutional layers
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
        
        // Pooling and dropout
        pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
        dropout = register_module("dropout", torch::nn::Dropout(0.5));
        
        // Fully connected layers
        fc1 = register_module("fc1", torch::nn::Linear(128 * 4 * 4, 512));
        fc2 = register_module("fc2", torch::nn::Linear(512, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Conv block 1: 32x32x3 -> 16x16x32
        x = pool(torch::relu(conv1(x)));
        
        // Conv block 2: 16x16x32 -> 8x8x64
        x = pool(torch::relu(conv2(x)));
        
        // Conv block 3: 8x8x64 -> 4x4x128
        x = pool(torch::relu(conv3(x)));
        
        // Flatten: 4x4x128 -> 2048
        x = x.view({x.size(0), -1});
        
        // Fully connected layers
        x = dropout(torch::relu(fc1(x)));
        x = fc2(x);
        
        return x;
    }
};

TORCH_MODULE(CNN);

int main() {
    // Create model for CIFAR-10 (10 classes)
    CNN model(10);
    
    // Create sample batch (batch_size=4, channels=3, height=32, width=32)
    auto input = torch::randn({4, 3, 32, 32});
    
    // Forward pass
    auto output = model->forward(input);
    
    // Apply softmax for probabilities
    auto probabilities = torch::softmax(output, 1);
    
    // Get predictions
    auto predictions = torch::argmax(probabilities, 1);
    
    std::cout << "Input shape: " << input.sizes() << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Predictions: " << predictions << std::endl;
    
    return 0;
}