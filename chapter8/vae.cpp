#include <torch/torch.h>
#include <iostream>
#include <tuple>

// Variational AutoEncoder implementation
struct VAE : torch::nn::Module {
    // Encoder layers
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    torch::nn::Linear fc_mu{nullptr}, fc_logvar{nullptr};
    
    // Decoder layers
    torch::nn::Linear fc_decode{nullptr};
    torch::nn::ConvTranspose2d deconv1{nullptr}, deconv2{nullptr}, deconv3{nullptr}, deconv4{nullptr};
    
    int latent_dim;
    
    VAE(int latent_dim = 128) : latent_dim(latent_dim) {
        // Encoder: 28x28 -> 1x1 feature map
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 4).stride(2).padding(1))); // 28->14
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 4).stride(2).padding(1))); // 14->7
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1))); // 7->3
        conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(0))); // 3->1
        
        // Latent space
        fc_mu = register_module("fc_mu", torch::nn::Linear(256, latent_dim));
        fc_logvar = register_module("fc_logvar", torch::nn::Linear(256, latent_dim));
        
        // Decoder
        fc_decode = register_module("fc_decode", torch::nn::Linear(latent_dim, 256));
        deconv1 = register_module("deconv1", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, 128, 3).stride(1).padding(0))); // 1->3
        deconv2 = register_module("deconv2", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1))); // 3->7
        deconv3 = register_module("deconv3", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 32, 4).stride(2).padding(1))); // 7->14
        deconv4 = register_module("deconv4", torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(32, 1, 4).stride(2).padding(1))); // 14->28
    }
    
    // Encoder forward pass
    std::tuple<torch::Tensor, torch::Tensor> encode(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));
        x = torch::relu(conv4->forward(x));
        
        // Flatten for fully connected layers
        x = x.view({x.size(0), -1});
        
        auto mu = fc_mu->forward(x);
        auto logvar = fc_logvar->forward(x);
        
        return std::make_tuple(mu, logvar);
    }
    
    // Reparameterization trick
    torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor logvar) {
        auto std = torch::exp(0.5 * logvar);
        auto eps = torch::randn_like(std);
        return mu + eps * std;
    }
    
    // Decoder forward pass
    torch::Tensor decode(torch::Tensor z) {
        auto x = torch::relu(fc_decode->forward(z));
        x = x.view({x.size(0), 256, 1, 1}); // Reshape for conv layers
        
        x = torch::relu(deconv1->forward(x));
        x = torch::relu(deconv2->forward(x));
        x = torch::relu(deconv3->forward(x));
        x = torch::sigmoid(deconv4->forward(x)); // Output in [0,1]
        
        return x;
    }
    
    // Full forward pass
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        auto [mu, logvar] = encode(x);
        auto z = reparameterize(mu, logvar);
        auto recon = decode(z);
        return std::make_tuple(recon, mu, logvar);
    }
    
    // Generate new samples
    torch::Tensor sample(int num_samples, torch::Device device) {
        auto z = torch::randn({num_samples, latent_dim}).to(device);
        return decode(z);
    }
    
    // Interpolate between two points in latent space
    torch::Tensor interpolate(torch::Tensor z1, torch::Tensor z2, int steps, torch::Device device) {
        std::vector<torch::Tensor> interpolations;
        for (int i = 0; i < steps; ++i) {
            float alpha = static_cast<float>(i) / (steps - 1);
            auto z_interp = (1 - alpha) * z1 + alpha * z2;
            interpolations.push_back(decode(z_interp));
        }
        return torch::cat(interpolations, 0);
    }
};

// VAE Loss function
torch::Tensor vae_loss(torch::Tensor recon_x, torch::Tensor x, torch::Tensor mu, torch::Tensor logvar, float beta = 1.0) {
    // Reconstruction loss (Binary Cross Entropy)
    auto recon_loss = torch::binary_cross_entropy(recon_x, x, {}, torch::Reduction::Sum);
    
    // KL divergence loss
    auto kl_loss = -0.5 * torch::sum(1 + logvar - mu.pow(2) - logvar.exp());
    
    return recon_loss + beta * kl_loss;
}

// Beta-VAE Loss with annealing
torch::Tensor beta_vae_loss(torch::Tensor recon_x, torch::Tensor x, torch::Tensor mu, torch::Tensor logvar, 
                           float beta, float capacity = 0.0, float gamma = 1000.0) {
    // Reconstruction loss
    auto recon_loss = torch::binary_cross_entropy(recon_x, x, {}, torch::Reduction::Sum);
    
    // KL divergence
    auto kl_div = -0.5 * torch::sum(1 + logvar - mu.pow(2) - logvar.exp());
    
    // Capacity constrained KL loss
    auto kl_loss = gamma * torch::abs(kl_div - capacity);
    
    return recon_loss + beta * kl_loss;
}
int main() {
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA" << std::endl;
    } else {
        std::cout << "Using CPU" << std::endl;
    }
    
    // Create VAE model
    int latent_dim = 128;
    VAE model(latent_dim);
    model.to(device);
    
    // Print model info
    std::cout << "=== Variational AutoEncoder ===" << std::endl;
    std::cout << "Latent dimension: " << latent_dim << std::endl;
    
    // Count parameters
    int total_params = 0;
    for (const auto& param : model.parameters()) {
        total_params += param.numel();
    }
    std::cout << "Total parameters: " << total_params << std::endl;
    
    // Optimizer
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    
    // Training parameters
    int batch_size = 64;
    int num_epochs = 100;
    float beta = 1.0; // Beta parameter for KL loss weighting
    
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Beta: " << beta << std::endl;
    std::cout << "Starting training..." << std::endl;
    
    // Training loop
    model.train();
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Generate dummy training data (normalized to [0,1])
        auto data = torch::rand({batch_size, 1, 28, 28}).to(device);
        
        optimizer.zero_grad();
        
        // Forward pass
        auto [recon_batch, mu, logvar] = model.forward(data);
        
        // Calculate VAE loss
        auto loss = vae_loss(recon_batch, data, mu, logvar, beta);
        
        // Backward pass
        loss.backward();
        optimizer.step();
        
        // Print progress
        if (epoch % 10 == 0) {
            auto recon_loss = torch::binary_cross_entropy(recon_batch, data, {}, torch::Reduction::Mean);
            auto kl_loss = -0.5 * torch::mean(1 + logvar - mu.pow(2) - logvar.exp());
            
            std::cout << "Epoch: " << epoch 
                      << ", Total Loss: " << loss.item<float>() / batch_size
                      << ", Recon Loss: " << recon_loss.item<float>()
                      << ", KL Loss: " << kl_loss.item<float>() << std::endl;
        }
    }
    
    std::cout << "Training completed!" << std::endl;
    
    // Switch to evaluation mode
    model.eval();
    torch::NoGradGuard no_grad;
    
    // Generate some samples
    std::cout << "\n=== Generating Samples ===" << std::endl;
    auto samples = model.sample(8, device);
    std::cout << "Generated " << samples.size(0) << " samples of size " 
              << samples.size(2) << "x" << samples.size(3) << std::endl;
    
    // Test reconstruction
    std::cout << "\n=== Testing Reconstruction ===" << std::endl;
    auto test_data = torch::rand({4, 1, 28, 28}).to(device);
    auto [recon_test, mu_test, logvar_test] = model.forward(test_data);
    auto recon_error = torch::mse_loss(recon_test, test_data);
    std::cout << "Reconstruction MSE: " << recon_error.item<float>() << std::endl;
    
    // Test interpolation
    std::cout << "\n=== Testing Interpolation ===" << std::endl;
    auto z1 = torch::randn({1, latent_dim}).to(device);
    auto z2 = torch::randn({1, latent_dim}).to(device);
    auto interpolated = model.interpolate(z1, z2, 5, device);
    std::cout << "Generated " << interpolated.size(0) << " interpolated samples" << std::endl;
    
    // Save model (optional)
    std::cout << "\n=== Saving Model ===" << std::endl;
    torch::save(model, "vae_model.pt");
    std::cout << "Model saved to vae_model.pt" << std::endl;
    
    return 0;
}