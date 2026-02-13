#include <torch/torch.h>
#include <iostream>

struct CNNAutoEncoder : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Conv2d bottleneck{nullptr};
    torch::nn::ConvTranspose2d deconv1{nullptr}, deconv2{nullptr}, deconv3{nullptr};
    
    CNNAutoEncoder() {
        conv1 = register_module("conv1", torch::nn::Conv2d(1, 32, 3));
        conv2 = register_module("conv2", torch::nn::Conv2d(32, 64, 3));
        conv3 = register_module("conv3", torch::nn::Conv2d(64, 128, 3));
        bottleneck = register_module("bottleneck", torch::nn::Conv2d(128, 64, 3));
        deconv1 = register_module("deconv1", torch::nn::ConvTranspose2d(64, 128, 3));
        deconv2 = register_module("deconv2", torch::nn::ConvTranspose2d(128, 64, 3));
        deconv3 = register_module("deconv3", torch::nn::ConvTranspose2d(64, 1, 3));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // Encoder (3 layers)
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv2->forward(x), 2));
        x = torch::relu(torch::max_pool2d(conv3->forward(x), 2));
        
        // Bottleneck (1 layer)
        x = torch::relu(bottleneck->forward(x));
        
        // Decoder (3 layers)
        x = torch::relu(torch::upsample_nearest2d(deconv1->forward(x), {14, 14}));
        x = torch::relu(torch::upsample_nearest2d(deconv2->forward(x), {56, 56}));
        x = torch::sigmoid(torch::upsample_nearest2d(deconv3->forward(x), {112, 112}));
        return x;
    }
};

int main() {
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA" << std::endl;
    }
    
    // Create model
    CNNAutoEncoder model;
    model.to(device);
    
    // Optimizer
    torch::optim::Adam optimizer(model.parameters(), 0.001);
    
    // Dummy training data (112x112 images)
    auto data = torch::randn({64, 1, 112, 112}).to(device);
    
    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        optimizer.zero_grad();
        auto output = model.forward(data);
        auto loss = torch::mse_loss(output, data);
        loss.backward();
        optimizer.step();
        
        if (epoch % 20 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<float>() << std::endl;
        }
    }
    
    return 0;
}