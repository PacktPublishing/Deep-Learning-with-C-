#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>

struct Seq2SeqRNNAutoEncoder : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::RNN encoder_rnn{nullptr};
    torch::nn::RNN decoder_rnn{nullptr};
    torch::nn::Linear output_projection{nullptr};
    
    int vocab_size;
    int embed_dim;
    int hidden_dim;
    int max_seq_len;
    int num_layers;
    
    Seq2SeqRNNAutoEncoder(int vocab_size = 10000, int embed_dim = 256, int hidden_dim = 512, 
                         int max_seq_len = 50, int num_layers = 2) 
        : vocab_size(vocab_size), embed_dim(embed_dim), hidden_dim(hidden_dim), 
          max_seq_len(max_seq_len), num_layers(num_layers) {
        
        // Shared embedding layer
        embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embed_dim));
        
        // Encoder RNN
        encoder_rnn = register_module("encoder_rnn", 
            torch::nn::RNN(torch::nn::RNNOptions(embed_dim, hidden_dim)
                .num_layers(num_layers)
                .batch_first(true)
                .nonlinearity(torch::kTanh)));
        
        // Decoder RNN
        decoder_rnn = register_module("decoder_rnn", 
            torch::nn::RNN(torch::nn::RNNOptions(embed_dim, hidden_dim)
                .num_layers(num_layers)
                .batch_first(true)
                .nonlinearity(torch::kTanh)));
        
        // Output projection to vocabulary
        output_projection = register_module("output_projection", torch::nn::Linear(hidden_dim, vocab_size));
    }
    
    // Encode input sequence to context vector
    torch::Tensor encode(torch::Tensor input_ids) {
        auto embedded = embedding->forward(input_ids);
        auto rnn_out = encoder_rnn->forward(embedded);
        auto output = std::get<0>(rnn_out);
        auto hidden_state = std::get<1>(rnn_out);
        
        // Return final hidden state as context
        return hidden_state;
    }
    
    // Decode from context vector to output sequence
    torch::Tensor decode(torch::Tensor target_ids, torch::Tensor hidden_state) {
        auto embedded = embedding->forward(target_ids);
        
        // Initialize decoder with encoder's final state
        auto rnn_out = decoder_rnn->forward(embedded, hidden_state);
        auto decoder_output = std::get<0>(rnn_out);
        
        // Project to vocabulary size
        auto logits = output_projection->forward(decoder_output);
        
        return logits;
    }
    
    // Full forward pass for training
    torch::Tensor forward(torch::Tensor source_ids, torch::Tensor target_ids) {
        // Encode source sequence
        auto hidden_state = encode(source_ids);
        
        // Decode to target sequence
        auto logits = decode(target_ids, hidden_state);
        
        return logits;
    }
    
    // Generate translation step by step (inference mode)
    torch::Tensor translate(torch::Tensor source_ids, int sos_token = 1, int eos_token = 2, torch::Device device = torch::kCPU) {
        torch::NoGradGuard no_grad;
        
        // Encode source
        auto hidden_state = encode(source_ids);
        
        // Start with SOS token
        auto current_token = torch::tensor({{sos_token}}, torch::kLong).to(device);
        std::vector<int64_t> generated_tokens;
        
        for (int i = 0; i < max_seq_len; ++i) {
            auto embedded = embedding->forward(current_token);
            
            auto rnn_out = decoder_rnn->forward(embedded, hidden_state);
            auto decoder_output = std::get<0>(rnn_out);
            auto new_hidden = std::get<1>(rnn_out);
            
            // Update hidden state for next step
            hidden_state = new_hidden;
            
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
    
    // Encode text to fixed-size representation (autoencoder bottleneck)
    torch::Tensor encode_to_vector(torch::Tensor input_ids) {
        auto hidden_state = encode(input_ids);
        // Take the last layer's hidden state as the encoded representation
        return hidden_state[-1];  // Shape: [batch_size, hidden_dim]
    }
    
    // Decode from fixed-size vector back to text
    torch::Tensor decode_from_vector(torch::Tensor encoded_vector, int max_len = 50, 
                                   int sos_token = 1, int eos_token = 2, torch::Device device = torch::kCPU) {
        torch::NoGradGuard no_grad;
        
        // Expand encoded vector to match RNN hidden state dimensions
        auto batch_size = encoded_vector.size(0);
        auto hidden_state = encoded_vector.unsqueeze(0).repeat({num_layers, 1, 1});
        
        auto current_token = torch::full({batch_size, 1}, sos_token, torch::kLong).to(device);
        std::vector<torch::Tensor> generated_sequence;
        
        for (int i = 0; i < max_len; ++i) {
            auto embedded = embedding->forward(current_token);
            
            auto rnn_out = decoder_rnn->forward(embedded, hidden_state);
            auto decoder_output = std::get<0>(rnn_out);
            auto new_hidden = std::get<1>(rnn_out);
            
            hidden_state = new_hidden;
            
            auto logits = output_projection->forward(decoder_output);
            auto next_token = torch::argmax(logits, -1);
            
            generated_sequence.push_back(next_token);
            current_token = next_token;
            
            // Check if all sequences in batch have generated EOS
            auto eos_mask = (next_token == eos_token);
            if (torch::all(eos_mask).item<bool>()) break;
        }
        
        return torch::cat(generated_sequence, 1);
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
    int num_layers = 2;
    int batch_size = 32;
    int seq_len = 20;
    
    // Create model
    Seq2SeqRNNAutoEncoder model(vocab_size, embed_dim, hidden_dim, max_seq_len, num_layers);
    model.to(device);
    
    std::cout << "=== Seq2Seq RNN AutoEncoder for Translation ===" << std::endl;
    std::cout << "Vocabulary size: " << vocab_size << std::endl;
    std::cout << "Embedding dimension: " << embed_dim << std::endl;
    std::cout << "Hidden dimension: " << hidden_dim << std::endl;
    std::cout << "Number of layers: " << num_layers << std::endl;
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
        auto loss = torch::nn::functional::cross_entropy(
            logits.view({-1, vocab_size}), 
            target_ids.view({-1})
        );
        
        // Backward pass
        loss.backward();
        
        // Gradient clipping for RNN stability
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
    
    // Test autoencoder functionality
    std::cout << "\n=== Testing Autoencoder Functionality ===" << std::endl;
    auto encoded_vector = model.encode_to_vector(test_source);
    std::cout << "Encoded vector shape: [" << encoded_vector.size(0) << ", " << encoded_vector.size(1) << "]" << std::endl;
    
    auto reconstructed = model.decode_from_vector(encoded_vector, 15, 1, 2, device);
    std::cout << "Reconstructed sequence: ";
    for (int i = 0; i < reconstructed.size(1); ++i) {
        std::cout << reconstructed[0][i].item<int64_t>() << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nSeq2Seq RNN AutoEncoder training and testing completed successfully!" << std::endl;
    
    return 0;
}