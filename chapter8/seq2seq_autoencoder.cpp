#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>

struct Seq2SeqAutoEncoder : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTM encoder_lstm{nullptr};
    torch::nn::LSTM decoder_lstm{nullptr};
    torch::nn::Linear output_projection{nullptr};
    
    int vocab_size;
    int embed_dim;
    int hidden_dim;
    int max_seq_len;
    
    Seq2SeqAutoEncoder(int vocab_size = 10000, int embed_dim = 256, int hidden_dim = 512, int max_seq_len = 50) 
        : vocab_size(vocab_size), embed_dim(embed_dim), hidden_dim(hidden_dim), max_seq_len(max_seq_len) {
        
        // Shared embedding layer
        embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embed_dim));
        
        // Encoder LSTM
        encoder_lstm = register_module("encoder_lstm", 
            torch::nn::LSTM(torch::nn::LSTMOptions(embed_dim, hidden_dim).batch_first(true)));
        
        // Decoder LSTM
        decoder_lstm = register_module("decoder_lstm", 
            torch::nn::LSTM(torch::nn::LSTMOptions(embed_dim, hidden_dim).batch_first(true)));
        
        // Output projection to vocabulary
        output_projection = register_module("output_projection", torch::nn::Linear(hidden_dim, vocab_size));
    }
    
    // Encode input sequence to context vector
    std::tuple<torch::Tensor, torch::Tensor> encode(torch::Tensor input_ids) {
        auto embedded = embedding->forward(input_ids);
        auto lstm_out = encoder_lstm->forward(embedded);
        auto output = std::get<0>(lstm_out);
        auto hidden_states = std::get<1>(lstm_out);
        
        // Return final hidden and cell states as context
        auto h_n = std::get<0>(hidden_states);  // Final hidden state
        auto c_n = std::get<1>(hidden_states);  // Final cell state
        
        return std::make_tuple(h_n, c_n);
    }
    
    // Decode from context vector to output sequence
    torch::Tensor decode(torch::Tensor target_ids, torch::Tensor hidden_state, torch::Tensor cell_state) {
        auto embedded = embedding->forward(target_ids);
        
        // Initialize decoder with encoder's final states
        std::tuple<torch::Tensor, torch::Tensor> initial_state = std::make_tuple(hidden_state, cell_state);
        
        auto lstm_out = decoder_lstm->forward(embedded, initial_state);
        auto decoder_output = std::get<0>(lstm_out);
        
        // Project to vocabulary size
        auto logits = output_projection->forward(decoder_output);
        
        return logits;
    }
    
    // Full forward pass for training
    torch::Tensor forward(torch::Tensor source_ids, torch::Tensor target_ids) {
        // Encode source sequence
        auto [hidden_state, cell_state] = encode(source_ids);
        
        // Decode to target sequence
        auto logits = decode(target_ids, hidden_state, cell_state);
        
        return logits;
    }
    
    // Generate translation (inference mode)
    torch::Tensor translate(torch::Tensor source_ids, int sos_token = 1, int eos_token = 2, torch::Device device = torch::kCPU) {
        torch::NoGradGuard no_grad;
        
        // Encode source
        auto [hidden_state, cell_state] = encode(source_ids);
        
        // Start with SOS token
        auto current_token = torch::tensor({{sos_token}}, torch::kLong).to(device);
        std::vector<int64_t> generated_tokens;
        
        for (int i = 0; i < max_seq_len; ++i) {
            auto embedded = embedding->forward(current_token);
            
            std::tuple<torch::Tensor, torch::Tensor> state = std::make_tuple(hidden_state, cell_state);
            auto lstm_out = decoder_lstm->forward(embedded, state);
            
            auto decoder_output = std::get<0>(lstm_out);
            auto new_states = std::get<1>(lstm_out);
            
            // Update states for next step
            hidden_state = std::get<0>(new_states);
            cell_state = std::get<1>(new_states);
            
            // Get next token
            auto logits = output_projection->forward(decoder_output);
            auto next_token = torch::argmax(logits, -1);
            
            int64_t token_id = next_token.item<int64_t>();
            generated_tokens.push_back(token_id);
            
            // Stop if EOS token generated
            if (token_id == eos_token) break;
            
            current_token = next_token;
        }
        
        return torch::tensor(generated_tokens, torch::kLong);
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
    int embed_dim = 256;
    int hidden_dim = 512;
    int max_seq_len = 50;
    int batch_size = 32;
    int seq_len = 20;
    
    // Create model
    Seq2SeqAutoEncoder model(vocab_size, embed_dim, hidden_dim, max_seq_len);
    model.to(device);
    
    std::cout << "=== Seq2Seq AutoEncoder for Translation ===" << std::endl;
    std::cout << "Vocabulary size: " << vocab_size << std::endl;
    std::cout << "Embedding dimension: " << embed_dim << std::endl;
    std::cout << "Hidden dimension: " << hidden_dim << std::endl;
    std::cout << "Max sequence length: " << max_seq_len << std::endl;
    
    // Count parameters
    int total_params = 0;
    for (const auto& param : model.parameters()) {
        total_params += param.numel();
    }
    std::cout << "Total parameters: " << total_params << std::endl;
    
    // Optimizer
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    
    std::cout << "\nStarting training..." << std::endl;
    
    // Training loop
    model.train();
    for (int epoch = 0; epoch < 100; ++epoch) {
        // Generate dummy training data (source and target sequences)
        auto source_ids = torch::randint(3, vocab_size, {batch_size, seq_len}, torch::kLong).to(device);
        auto target_ids = torch::randint(3, vocab_size, {batch_size, seq_len}, torch::kLong).to(device);
        
        optimizer.zero_grad();
        
        // Forward pass
        auto logits = model.forward(source_ids, target_ids);
        
        // Calculate cross-entropy loss
        auto loss = torch::cross_entropy(
            logits.view({-1, vocab_size}), 
            target_ids.view({-1})
        );
        
        // Backward pass
        loss.backward();
        
        // Gradient clipping for stability
        torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
        
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
    
    // Switch to evaluation mode
    model.eval();
    
    // Test translation
    std::cout << "\n=== Testing Translation ===" << std::endl;
    auto test_source = torch::randint(3, vocab_size, {1, 10}, torch::kLong).to(device);
    
    std::cout << "Source sequence: ";
    for (int i = 0; i < test_source.size(1); ++i) {
        std::cout << test_source[0][i].item<int64_t>() << " ";
    }
    std::cout << std::endl;
    
    auto translation = model.translate(test_source, 1, 2, device);
    
    std::cout << "Generated translation: ";
    for (int i = 0; i < translation.size(0); ++i) {
        std::cout << translation[i].item<int64_t>() << " ";
    }
    std::cout << std::endl;
    
    // Save model
    std::cout << "\n=== Saving Model ===" << std::endl;
    torch::save(model, "seq2seq_autoencoder.pt");
    std::cout << "Model saved to seq2seq_autoencoder.pt" << std::endl;
    
    return 0;
}