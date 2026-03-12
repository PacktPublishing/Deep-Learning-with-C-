#include <torch/torch.h>
class UNetImpl : public torch::nn::Module {
private:
    // Encoder
    torch::nn::Sequential encoder1{nullptr}, encoder2{nullptr}, encoder3{nullptr}, encoder4{nullptr};
    torch::nn::MaxPool2d maxpool{nullptr};
    
    // Decoder
    torch::nn::Sequential decoder4{nullptr}, decoder3{nullptr}, decoder2{nullptr}, decoder1{nullptr};
    torch::nn::ConvTranspose2d upconv4{nullptr}, upconv3{nullptr}, upconv2{nullptr}, upconv1{nullptr};
    
    // Final layer
    torch::nn::Conv2d final_layer{nullptr};

public:
    UNetImpl(int in_channels = 1, int out_channels = 1) {
        // Encoder blocks
        encoder1 = register_module("encoder1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        ));
 
        encoder2 = register_module("encoder2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        ));
 
        encoder3 = register_module("encoder3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        ));
 
        encoder4 = register_module("encoder4", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        ));
 
        maxpool = register_module("maxpool", torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions(2).stride(2)));
 
        // Decoder blocks with up-convolutions
        upconv4 = register_module("upconv4", 
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(512, 256, 2).stride(2)));
        upconv3 = register_module("upconv3", 
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 2).stride(2)));
        upconv2 = register_module("upconv2", 
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 2).stride(2)));
 
        decoder4 = register_module("decoder4", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        ));
 
        decoder3 = register_module("decoder3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        ));
 
        decoder2 = register_module("decoder2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        ));
 
        final_layer = register_module("final", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, out_channels, 1)));
    }
 
    torch::Tensor forward(torch::Tensor x) {
        // Encoder path
        auto e1 = encoder1->forward(x);
        auto e2 = encoder2->forward(maxpool(e1));
        auto e3 = encoder3->forward(maxpool(e2));
        auto e4 = encoder4->forward(maxpool(e3));
 
        // Decoder path without skip connections
        auto d4 = decoder4->forward(upconv4(e4));
        auto d3 = decoder3->forward(upconv3(d4));
        auto d2 = decoder2->forward(upconv2(d3));
 
        return torch::sigmoid(final_layer(d2));
    }
};
 
TORCH_MODULE(UNet);
 
// Usage example
int main() {
    // Create model
    UNet model(1, 1);
    
    // Create sample input (batch_size, channels, height, width)
    auto input = torch::randn({1, 1, 256, 256});
    
    // Forward pass
    auto output = model->forward(input);
    
    std::cout << "Input shape: " << input.sizes() << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Output min: " << output.min().item<float>() << std::endl;
    std::cout << "Output max: " << output.max().item<float>() << std::endl;
    
    return 0;
}

