#include <torch/torch.h>
#include <iostream>
#include <memory>

// Define the network structure following PyTorch C++ frontend patterns
struct ConvNetImpl : torch::nn::Module {
    ConvNetImpl(int64_t input_channels, int64_t output_channels, int64_t kernel_size)
        : conv1(torch::nn::Conv2dOptions(input_channels, output_channels, kernel_size).padding(0)),
          relu() {
        
        // Register modules
        register_module("conv1", conv1);
        register_module("relu", relu);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = conv1(x);
        x = relu(x);
        return x;
    }
    
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::ReLU relu{nullptr};
};

// Create module holder following tutorial pattern
TORCH_MODULE(ConvNet);

int main() {
    torch::manual_seed(1);
    
    // Device detection following tutorial pattern
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }
    
    // Create network using module holder
    ConvNet net(1, 64, 3);
    net->to(device);
    
    // Initialize weights following tutorial pattern
    net->apply([](torch::nn::Module& module) {
        torch::NoGradGuard no_grad;
        if (auto* conv = module.as<torch::nn::Conv2d>()) {
            torch::nn::init::uniform_(conv->weight, -0.1, 0.1);
            torch::nn::init::zeros_(conv->bias);
        }
    });
    
    // Create input tensor following tutorial pattern
    auto input = torch::arange(1, 65, torch::dtype(torch::kFloat32).device(device))
                     .reshape({1, 1, 8, 8});
    
    // Forward pass
    torch::NoGradGuard no_grad;
    auto output = net->forward(input);
    
    std::cout << "Input shape: " << input.sizes() << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "LibTorch convolution completed successfully!" << std::endl;
    std::cout << "Using device: " << device << std::endl;
    
    return 0;
}