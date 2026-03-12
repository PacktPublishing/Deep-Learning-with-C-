#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>

// Simple Transformer-based LLM for autoregressive generation
struct AutoregressiveLLM : torch::nn::Module {
    torch::nn::Embedding token_embedding{nullptr};
    torch::nn::Embedding position_embedding{nullptr};
    torch::nn::TransformerEncoder transformer{nullptr};
    torch::nn::Linear output_projection{nullptr};
    
    int vocab_size;
    int embed_dim;
    int max_seq_len;
    int num_heads;
    int num_layers;
    
    AutoregressiveLLM(int vocab_size = 10000, int embed_dim = 512, int max_seq_len = 128, 
                     int num_heads = 8, int num_layers = 6) 
        : vocab_size(vocab_size), embed_dim(embed_dim), max_seq_len(max_seq_len),
          num_heads(num_heads), num_layers(num_layers) {
        
        // Token and position embeddings
        token_embedding = register_module("token_embedding", torch::nn::Embedding(vocab_size, embed_dim));
        position_embedding = register_module("position_embedding", torch::nn::Embedding(max_seq_len, embed_dim));
        
        // Transformer encoder layers
        auto encoder_layer = torch::nn::TransformerEncoderLayer(
            torch::nn::TransformerEncoderLayerOptions(embed_dim, num_heads)
                .dim_feedforward(embed_dim * 4)
                .dropout(0.1));
        
        transformer = register_module("transformer", 
            torch::nn::TransformerEncoder(encoder_layer, num_layers));
        
        // Output projection to vocabulary
        output_projection = register_module("output_projection", torch::nn::Linear(embed_dim, vocab_size));
    }
    
    // Create causal mask for autoregressive generation
    torch::Tensor create_causal_mask(int seq_len, torch::Device device) {
        auto mask = torch::triu(torch::ones({seq_len, seq_len}, torch::kBool), 1).to(device);
        return mask;
    }
    
    // Forward pass with causal masking
    torch::Tensor forward(torch::Tensor input_ids) {
        auto seq_len = input_ids.size(1);
        auto device = input_ids.device();
        
        // Create position indices
        auto positions = torch::arange(seq_len, torch::kLong).unsqueeze(0).to(device);
        
        // Embeddings
        auto token_embeds = token_embedding->forward(input_ids);
        auto pos_embeds = position_embedding->forward(positions);
        auto embeddings = token_embeds + pos_embeds;
        
        // Transpose for transformer (seq_len, batch, embed_dim)
        embeddings = embeddings.transpose(0, 1);
        
        // Apply causal mask
        auto causal_mask = create_causal_mask(seq_len, device);
        
        // Transformer forward pass
        auto transformer_output = transformer->forward(embeddings, causal_mask);
        
        // Transpose back (batch, seq_len, embed_dim)
        transformer_output = transformer_output.transpose(0, 1);
        
        // Project to vocabulary
        auto logits = output_projection->forward(transformer_output);
        
        return logits;
    }
    
    // Autoregressive text generation
    std::vector<int64_t> generate(std::vector<int64_t> prompt, int max_new_tokens = 50, 
                                 float temperature = 1.0, torch::Device device = torch::kCPU) {
        torch::NoGradGuard no_grad;
        eval();
        
        std::vector<int64_t> generated_sequence = prompt;
        
        for (int i = 0; i < max_new_tokens; ++i) {
            // Convert current sequence to tensor
            auto input_tensor = torch::from_blob(
                generated_sequence.data(), 
                {1, static_cast<long>(generated_sequence.size())}, 
                torch::kLong
            ).clone().to(device);
            
            // Get logits for next token
            auto logits = forward(input_tensor);
            
            // Extract logits for last position
            auto next_token_logits = logits[0][-1] / temperature;
            
            // Sample next token (using multinomial sampling)
            auto probs = torch::softmax(next_token_logits, -1);
            auto next_token = torch::multinomial(probs, 1);
            
            int64_t token_id = next_token.item<int64_t>();
            generated_sequence.push_back(token_id);
            
            // Stop if sequence gets too long
            if (generated_sequence.size() >= max_seq_len) break;
        }
        
        return generated_sequence;
    }
    
    // Greedy generation (deterministic)
    std::vector<int64_t> generate_greedy(std::vector<int64_t> prompt, int max_new_tokens = 50,
                                        torch::Device device = torch::kCPU) {
        torch::NoGradGuard no_grad;
        eval();
        
        std::vector<int64_t> generated_sequence = prompt;
        
        for (int i = 0; i < max_new_tokens; ++i) {
            auto input_tensor = torch::from_blob(
                generated_sequence.data(), 
                {1, static_cast<long>(generated_sequence.size())}, 
                torch::kLong
            ).clone().to(device);
            
            auto logits = forward(input_tensor);
            
            // Take most likely next token
            auto next_token = torch::argmax(logits[0][-1], -1);
            int64_t token_id = next_token.item<int64_t>();
            
            generated_sequence.push_back(token_id);
            
            if (generated_sequence.size() >= max_seq_len) break;
        }
        
        return generated_sequence;
    }
};

int main() {
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA" << std::endl;
    } else {
        std::cout << "Using CPU" << std::endl;
    }
    
    // Model parameters
    int vocab_size = 10000;
    int embed_dim = 512;
    int max_seq_len = 128;
    int num_heads = 8;
    int num_layers = 6;
    int batch_size = 32;
    int seq_len = 64;
    
    // Create LLM
    AutoregressiveLLM llm(vocab_size, embed_dim, max_seq_len, num_heads, num_layers);
    llm.to(device);
    
    std::cout << "=== Autoregressive LLM ===" << std::endl;
    std::cout << "Vocabulary size: " << vocab_size << std::endl;
    std::cout << "Embedding dimension: " << embed_dim << std::endl;
    std::cout << "Max sequence length: " << max_seq_len << std::endl;
    std::cout << "Number of attention heads: " << num_heads << std::endl;
    std::cout << "Number of layers: " << num_layers << std::endl;
    
    // Count parameters
    int total_params = 0;
    for (const auto& param : llm.parameters()) {
        total_params += param.numel();
    }
    std::cout << "Total parameters: " << total_params << std::endl;
    
    // Optimizer
    torch::optim::Adam optimizer(llm.parameters(), torch::optim::AdamOptions(1e-4));
    
    std::cout << "\nStarting autoregressive training..." << std::endl;
    
    // Training loop
    llm.train();
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Generate dummy training data
        auto input_ids = torch::randint(1, vocab_size, {batch_size, seq_len}, torch::kLong).to(device);
        auto target_ids = torch::randint(1, vocab_size, {batch_size, seq_len}, torch::kLong).to(device);
        
        optimizer.zero_grad();
        
        // Forward pass
        auto logits = llm.forward(input_ids);
        
        // Autoregressive loss: predict next token at each position
        auto loss = torch::nn::functional::cross_entropy(
            logits.view({-1, vocab_size}), 
            target_ids.view({-1})
        );
        
        // Backward pass
        loss.backward();
        
        // Gradient clipping
        torch::nn::utils::clip_grad_norm_(llm.parameters(), 1.0);
        
        optimizer.step();
        
        // Print progress
        if (epoch % 20 == 0) {
            auto perplexity = torch::exp(loss);
            std::cout << "Epoch: " << epoch 
                      << ", Loss: " << loss.item<float>()
                      << ", Perplexity: " << perplexity.item<float>() << std::endl;
        }
    }
    
    std::cout << "Training completed!" << std::endl;
    
    // Test autoregressive generation
    std::cout << "\n=== Testing Autoregressive Generation ===" << std::endl;
    
    // Create a prompt
    std::vector<int64_t> prompt = {1, 42, 123, 456}; // Example token IDs
    
    std::cout << "Prompt tokens: ";
    for (auto token : prompt) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    // Greedy generation
    auto greedy_output = llm.generate_greedy(prompt, 20, device);
    std::cout << "Greedy generation: ";
    for (size_t i = prompt.size(); i < greedy_output.size(); ++i) {
        std::cout << greedy_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Sampling generation
    auto sampled_output = llm.generate(prompt, 20, 0.8, device);
    std::cout << "Sampled generation (T=0.8): ";
    for (size_t i = prompt.size(); i < sampled_output.size(); ++i) {
        std::cout << sampled_output[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nAutoregressive LLM training and testing completed successfully!" << std::endl;
    
    return 0;
}