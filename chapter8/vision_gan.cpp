#include <torch/torch.h>
#include <iostream>
#include <vector>

// Generator Network
struct Generator : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    torch::nn::ConvTranspose2d deconv1{nullptr}, deconv2{nullptr}, deconv3{nullptr}, deconv4{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    
    int latent_dim;
    int img_channels;
    
    Generator(int latent_dim = 100, int img_channels = 3) : latent_dim(latent_dim), img_channels(img_channels) {
        // Project latent vector to feature map
        fc1 = register_module("fc1", torch::nn::Linear(latent_dim, 512 * 4 * 4));
        
        // Transposed convolutions for upsampling
        deconv1 = register_module("deconv1", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(512, 256, 4).stride(2).padding(1))); // 4x4 -> 8x8
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(256));
        
        deconv2 = register_module("deconv2", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1))); // 8x8 -> 16x16
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
        
        deconv3 = register_module("deconv3", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1))); // 16x16 -> 32x32
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
        
        deconv4 = register_module("deconv4", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(64, img_channels, 4).stride(2).padding(1))); // 32x32 -> 64x64
    }
    
    torch::Tensor forward(torch::Tensor z) {
        // Project and reshape
        auto x = torch::relu(fc1->forward(z));
        x = x.view({x.size(0), 512, 4, 4});
        
        // Upsampling layers
        x = torch::relu(bn1->forward(deconv1->forward(x)));
        x = torch::relu(bn2->forward(deconv2->forward(x)));
        x = torch::relu(bn3->forward(deconv3->forward(x)));
        x = torch::tanh(deconv4->forward(x)); // Output in [-1, 1]
        
        return x;
    }
};

// Discriminator Network
struct Discriminator : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Linear fc1{nullptr};
    torch::nn::Dropout dropout{nullptr};
    
    int img_channels;
    
    Discriminator(int img_channels = 3) : img_channels(img_channels) {
        // Convolutional layers for downsampling
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(img_channels, 64, 4).stride(2).padding(1))); // 64x64 -> 32x32
        
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1))); // 32x32 -> 16x16
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(128));
        
        conv3 = register_module("conv3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1))); // 16x16 -> 8x8
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(256));
        
        conv4 = register_module("conv4", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(256, 512, 4).stride(2).padding(1))); // 8x8 -> 4x4
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(512));
        
        // Final classification layer
        fc1 = register_module("fc1", torch::nn::Linear(512 * 4 * 4, 1));
        dropout = register_module("dropout", torch::nn::Dropout(0.3));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // Downsampling layers
        x = torch::leaky_relu(conv1->forward(x), 0.2);
        x = torch::leaky_relu(bn1->forward(conv2->forward(x)), 0.2);
        x = torch::leaky_relu(bn2->forward(conv3->forward(x)), 0.2);
        x = torch::leaky_relu(bn3->forward(conv4->forward(x)), 0.2);
        
        // Flatten and classify
        x = x.view({x.size(0), -1});
        x = dropout->forward(x);
        x = torch::sigmoid(fc1->forward(x));
        
        return x;
    }
};

// Vision GAN class combining Generator and Discriminator
struct VisionGAN : torch::nn::Module {
    std::shared_ptr<Generator> generator;
    std::shared_ptr<Discriminator> discriminator;
    
    int latent_dim;
    int img_channels;
    
    VisionGAN(int latent_dim = 100, int img_channels = 3) 
        : latent_dim(latent_dim), img_channels(img_channels) {
        
        generator = register_module("generator", std::make_shared<Generator>(latent_dim, img_channels));
        discriminator = register_module("discriminator", std::make_shared<Discriminator>(img_channels));
    }
    
    // Generate images from noise
    torch::Tensor generate(int num_samples, torch::Device device) {
        torch::NoGradGuard no_grad;
        auto noise = torch::randn({num_samples, latent_dim}).to(device);
        return generator->forward(noise);
    }
    
    // Interpolate between two noise vectors
    torch::Tensor interpolate(torch::Tensor z1, torch::Tensor z2, int steps, torch::Device device) {
        torch::NoGradGuard no_grad;
        std::vector<torch::Tensor> interpolations;
        
        for (int i = 0; i < steps; ++i) {
            float alpha = static_cast<float>(i) / (steps - 1);
            auto z_interp = (1 - alpha) * z1 + alpha * z2;
            interpolations.push_back(generator->forward(z_interp));
        }
        
        return torch::cat(interpolations, 0);
    }
};

// GAN Loss functions
torch::Tensor adversarial_loss(torch::Tensor pred, torch::Tensor target) {
    return torch::binary_cross_entropy(pred, target);
}

int main() {
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA" << std::endl;
    } else {
        std::cout << "Using CPU" << std::endl;
    }
    
    // Model parameters
    int latent_dim = 100;
    int img_channels = 3;
    int img_size = 64;
    int batch_size = 64;
    int num_epochs = 200;
    float lr = 0.0002;
    
    // Create GAN
    VisionGAN gan(latent_dim, img_channels);
    gan.to(device);
    
    std::cout << "=== Vision GAN ===" << std::endl;
    std::cout << "Latent dimension: " << latent_dim << std::endl;
    std::cout << "Image channels: " << img_channels << std::endl;
    std::cout << "Image size: " << img_size << "x" << img_size << std::endl;
    
    // Count parameters
    int gen_params = 0, disc_params = 0;
    for (const auto& param : gan.generator->parameters()) {
        gen_params += param.numel();
    }
    for (const auto& param : gan.discriminator->parameters()) {
        disc_params += param.numel();
    }
    std::cout << "Generator parameters: " << gen_params << std::endl;
    std::cout << "Discriminator parameters: " << disc_params << std::endl;
    
    // Optimizers
    torch::optim::Adam gen_optimizer(gan.generator->parameters(), 
        torch::optim::AdamOptions(lr).betas(std::make_tuple(0.5, 0.999)));
    torch::optim::Adam disc_optimizer(gan.discriminator->parameters(), 
        torch::optim::AdamOptions(lr).betas(std::make_tuple(0.5, 0.999)));
    
    std::cout << "\nStarting GAN training..." << std::endl;
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Generate dummy real images (normalized to [-1, 1])
        auto real_images = torch::randn({batch_size, img_channels, img_size, img_size}).to(device);
        real_images = torch::tanh(real_images); // Simulate real data distribution
        
        auto real_labels = torch::ones({batch_size, 1}).to(device);
        auto fake_labels = torch::zeros({batch_size, 1}).to(device);
        
        // Train Discriminator
        disc_optimizer.zero_grad();
        
        // Real images
        auto real_pred = gan.discriminator->forward(real_images);
        auto real_loss = adversarial_loss(real_pred, real_labels);
        
        // Fake images
        auto noise = torch::randn({batch_size, latent_dim}).to(device);
        auto fake_images = gan.generator->forward(noise);
        auto fake_pred = gan.discriminator->forward(fake_images.detach());
        auto fake_loss = adversarial_loss(fake_pred, fake_labels);
        
        auto disc_loss = (real_loss + fake_loss) / 2;
        disc_loss.backward();
        disc_optimizer.step();
        
        // Train Generator
        gen_optimizer.zero_grad();
        
        auto gen_noise = torch::randn({batch_size, latent_dim}).to(device);
        auto gen_images = gan.generator->forward(gen_noise);
        auto gen_pred = gan.discriminator->forward(gen_images);
        auto gen_loss = adversarial_loss(gen_pred, real_labels); // Generator wants to fool discriminator
        
        gen_loss.backward();
        gen_optimizer.step();
        
        // Print progress
        if (epoch % 20 == 0) {
            std::cout << "Epoch: " << epoch 
                      << ", D Loss: " << disc_loss.item<float>()
                      << ", G Loss: " << gen_loss.item<float>()
                      << ", D(real): " << torch::mean(real_pred).item<float>()
                      << ", D(fake): " << torch::mean(fake_pred).item<float>() << std::endl;
        }
    }
    
    std::cout << "Training completed!" << std::endl;
    
    // Switch to evaluation mode
    gan.eval();
    
    // Generate samples
    std::cout << "\n=== Generating Samples ===" << std::endl;
    auto generated_images = gan.generate(8, device);
    std::cout << "Generated " << generated_images.size(0) << " images of size " 
              << generated_images.size(2) << "x" << generated_images.size(3) << std::endl;
    
    // Test interpolation
    std::cout << "\n=== Testing Interpolation ===" << std::endl;
    auto z1 = torch::randn({1, latent_dim}).to(device);
    auto z2 = torch::randn({1, latent_dim}).to(device);
    auto interpolated = gan.interpolate(z1, z2, 5, device);
    std::cout << "Generated " << interpolated.size(0) << " interpolated images" << std::endl;
    
    std::cout << "\nVision GAN training and testing completed successfully!" << std::endl;
    
    return 0;
}