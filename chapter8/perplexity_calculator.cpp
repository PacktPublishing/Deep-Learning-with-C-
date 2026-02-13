#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>

// Simple Language Model for perplexity calculation
struct LanguageModel : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear output{nullptr};
    
    int vocab_size;
    int embed_dim;
    int hidden_dim;
    
    LanguageModel(int vocab_size = 10000, int embed_dim = 256, int hidden_dim = 512) 
        : vocab_size(vocab_size), embed_dim(embed_dim), hidden_dim(hidden_dim) {
        
        embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embed_dim));
        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(embed_dim, hidden_dim).batch_first(true)));
        output = register_module("output", torch::nn::Linear(hidden_dim, vocab_size));
    }
    
    torch::Tensor forward(torch::Tensor input_ids) {
        auto embedded = embedding->forward(input_ids);
        auto lstm_out = lstm->forward(embedded);
        auto hidden_states = std::get<0>(lstm_out);
        return output->forward(hidden_states);
    }
};

class PerplexityCalculator {
private:
    LanguageModel& model;
    torch::Device device;
    
public:
    PerplexityCalculator(LanguageModel& m, torch::Device d) : model(m), device(d) {}
    
    // Calculate perplexity on a single sequence
    float calculate_sequence_perplexity(std::vector<int64_t> sequence) {
        torch::NoGradGuard no_grad;
        model.eval();
        
        if (sequence.size() < 2) {
            std::cerr << "Sequence too short for perplexity calculation" << std::endl;
            return std::numeric_limits<float>::infinity();
        }
        
        // Prepare input and target
        auto input_ids = std::vector<int64_t>(sequence.begin(), sequence.end() - 1);
        auto target_ids = std::vector<int64_t>(sequence.begin() + 1, sequence.end());
        
        auto input_tensor = torch::tensor({input_ids}, torch::kLong).to(device);
        auto target_tensor = torch::tensor(target_ids, torch::kLong).to(device);
        
        // Forward pass
        auto logits = model.forward(input_tensor);
        logits = logits.squeeze(0); // Remove batch dimension
        
        // Calculate cross-entropy loss
        auto loss = torch::cross_entropy(logits, target_tensor, {}, torch::Reduction::Mean);
        
        // Perplexity = exp(loss)
        return torch::exp(loss).item<float>();
    }
    
    // Calculate perplexity on multiple sequences (batch)
    float calculate_batch_perplexity(std::vector<std::vector<int64_t>> sequences) {
        torch::NoGradGuard no_grad;
        model.eval();
        
        float total_log_likelihood = 0.0f;
        int total_tokens = 0;
        
        for (const auto& sequence : sequences) {
            if (sequence.size() < 2) continue;
            
            // Prepare input and target
            auto input_ids = std::vector<int64_t>(sequence.begin(), sequence.end() - 1);
            auto target_ids = std::vector<int64_t>(sequence.begin() + 1, sequence.end());
            
            auto input_tensor = torch::tensor({input_ids}, torch::kLong).to(device);
            auto target_tensor = torch::tensor(target_ids, torch::kLong).to(device);
            
            // Forward pass
            auto logits = model.forward(input_tensor);
            logits = logits.squeeze(0);
            
            // Calculate log probabilities
            auto log_probs = torch::log_softmax(logits, -1);
            
            // Sum log probabilities for target tokens
            for (size_t i = 0; i < target_ids.size(); ++i) {
                total_log_likelihood += log_probs[i][target_ids[i]].item<float>();
                total_tokens++;
            }
        }
        
        // Average negative log likelihood
        float avg_nll = -total_log_likelihood / total_tokens;
        
        // Perplexity = exp(average negative log likelihood)
        return std::exp(avg_nll);
    }
    
    // Calculate perplexity with detailed token-level analysis
    struct TokenPerplexity {
        int64_t token_id;
        float probability;
        float surprise; // -log(probability)
    };
    
    std::pair<float, std::vector<TokenPerplexity>> calculate_detailed_perplexity(std::vector<int64_t> sequence) {
        torch::NoGradGuard no_grad;
        model.eval();
        
        std::vector<TokenPerplexity> token_details;
        
        if (sequence.size() < 2) {
            return {std::numeric_limits<float>::infinity(), token_details};
        }
        
        auto input_ids = std::vector<int64_t>(sequence.begin(), sequence.end() - 1);
        auto target_ids = std::vector<int64_t>(sequence.begin() + 1, sequence.end());
        
        auto input_tensor = torch::tensor({input_ids}, torch::kLong).to(device);
        auto logits = model.forward(input_tensor);
        logits = logits.squeeze(0);
        
        auto probs = torch::softmax(logits, -1);
        
        float total_surprise = 0.0f;
        
        for (size_t i = 0; i < target_ids.size(); ++i) {
            int64_t token_id = target_ids[i];
            float prob = probs[i][token_id].item<float>();
            float surprise = -std::log(prob);
            
            token_details.push_back({token_id, prob, surprise});
            total_surprise += surprise;
        }
        
        float avg_surprise = total_surprise / target_ids.size();
        float perplexity = std::exp(avg_surprise);
        
        return {perplexity, token_details};
    }
    
    // Calculate perplexity for different context lengths
    std::vector<float> calculate_context_perplexity(std::vector<int64_t> sequence, 
                                                   std::vector<int> context_lengths) {
        torch::NoGradGuard no_grad;
        model.eval();
        
        std::vector<float> perplexities;
        
        for (int context_len : context_lengths) {
            if (context_len >= static_cast<int>(sequence.size())) {
                perplexities.push_back(std::numeric_limits<float>::infinity());
                continue;
            }
            
            // Use only the specified context length
            auto context_seq = std::vector<int64_t>(
                sequence.end() - context_len - 1, sequence.end());
            
            float perplexity = calculate_sequence_perplexity(context_seq);
            perplexities.push_back(perplexity);
        }
        
        return perplexities;
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
    
    // Create language model
    int vocab_size = 1000;
    int embed_dim = 256;
    int hidden_dim = 512;
    
    LanguageModel model(vocab_size, embed_dim, hidden_dim);
    model.to(device);
    
    std::cout << "=== Perplexity Calculator ===" << std::endl;
    std::cout << "Vocabulary size: " << vocab_size << std::endl;
    
    // Initialize perplexity calculator
    PerplexityCalculator calculator(model, device);
    
    // Test sequences
    std::vector<int64_t> test_sequence1 = {1, 42, 123, 456, 789, 234, 567, 890};
    std::vector<int64_t> test_sequence2 = {2, 100, 200, 300, 400, 500};
    std::vector<int64_t> test_sequence3 = {3, 50, 150, 250, 350};
    
    std::cout << "\n=== Single Sequence Perplexity ===" << std::endl;
    
    // Calculate perplexity for individual sequences
    float perp1 = calculator.calculate_sequence_perplexity(test_sequence1);
    float perp2 = calculator.calculate_sequence_perplexity(test_sequence2);
    float perp3 = calculator.calculate_sequence_perplexity(test_sequence3);
    
    std::cout << "Sequence 1 perplexity: " << perp1 << std::endl;
    std::cout << "Sequence 2 perplexity: " << perp2 << std::endl;
    std::cout << "Sequence 3 perplexity: " << perp3 << std::endl;
    
    // Batch perplexity
    std::cout << "\n=== Batch Perplexity ===" << std::endl;
    std::vector<std::vector<int64_t>> batch_sequences = {test_sequence1, test_sequence2, test_sequence3};
    float batch_perp = calculator.calculate_batch_perplexity(batch_sequences);
    std::cout << "Batch perplexity: " << batch_perp << std::endl;
    
    // Detailed token-level analysis
    std::cout << "\n=== Detailed Token Analysis ===" << std::endl;
    auto [detailed_perp, token_details] = calculator.calculate_detailed_perplexity(test_sequence1);
    std::cout << "Detailed perplexity: " << detailed_perp << std::endl;
    std::cout << "Token-level analysis:" << std::endl;
    
    for (size_t i = 0; i < std::min(token_details.size(), size_t(5)); ++i) {
        const auto& detail = token_details[i];
        std::cout << "  Token " << detail.token_id 
                  << ": prob=" << detail.probability 
                  << ", surprise=" << detail.surprise << std::endl;
    }
    
    // Context length analysis
    std::cout << "\n=== Context Length Analysis ===" << std::endl;
    std::vector<int> context_lengths = {2, 3, 4, 5};
    auto context_perps = calculator.calculate_context_perplexity(test_sequence1, context_lengths);
    
    for (size_t i = 0; i < context_lengths.size(); ++i) {
        std::cout << "Context length " << context_lengths[i] 
                  << ": perplexity = " << context_perps[i] << std::endl;
    }
    
    std::cout << "\n=== Perplexity Interpretation ===" << std::endl;
    std::cout << "Lower perplexity = better model performance" << std::endl;
    std::cout << "Perplexity ≈ average branching factor" << std::endl;
    std::cout << "Perfect model: perplexity = 1.0" << std::endl;
    std::cout << "Random model: perplexity ≈ vocabulary_size" << std::endl;
    
    return 0;
}