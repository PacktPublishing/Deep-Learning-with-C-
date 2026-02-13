#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>

// Beam search candidate
struct BeamCandidate {
    std::vector<int64_t> sequence;
    float score;
    
    BeamCandidate(std::vector<int64_t> seq, float s) : sequence(seq), score(s) {}
    
    bool operator<(const BeamCandidate& other) const {
        return score < other.score; // For max heap
    }
};

// Simple LLM for demonstration
struct SimpleLLM : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, output{nullptr};
    
    int vocab_size;
    int embed_dim;
    
    SimpleLLM(int vocab_size = 10000, int embed_dim = 256) 
        : vocab_size(vocab_size), embed_dim(embed_dim) {
        
        embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embed_dim));
        fc1 = register_module("fc1", torch::nn::Linear(embed_dim, embed_dim));
        fc2 = register_module("fc2", torch::nn::Linear(embed_dim, embed_dim));
        output = register_module("output", torch::nn::Linear(embed_dim, vocab_size));
    }
    
    torch::Tensor forward(torch::Tensor input_ids) {
        auto x = embedding->forward(input_ids);
        x = torch::mean(x, 1); // Simple pooling
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return output->forward(x);
    }
};

class TextGenerator {
private:
    SimpleLLM& model;
    torch::Device device;
    
public:
    TextGenerator(SimpleLLM& m, torch::Device d) : model(m), device(d) {}
    
    // 1. Greedy Search - Always pick most likely token
    std::vector<int64_t> greedy_search(std::vector<int64_t> prompt, int max_length = 50) {
        torch::NoGradGuard no_grad;
        model.eval();
        
        std::vector<int64_t> sequence = prompt;
        
        for (int i = 0; i < max_length; ++i) {
            auto input_tensor = torch::tensor({sequence}, torch::kLong).to(device);
            auto logits = model.forward(input_tensor);
            
            // Get most likely next token
            auto next_token = torch::argmax(logits, -1);
            int64_t token_id = next_token.item<int64_t>();
            
            sequence.push_back(token_id);
        }
        
        return sequence;
    }
    
    // 2. Beam Search - Keep top k candidates
    std::vector<int64_t> beam_search(std::vector<int64_t> prompt, int beam_width = 3, int max_length = 50) {
        torch::NoGradGuard no_grad;
        model.eval();
        
        // Initialize beam with prompt
        std::priority_queue<BeamCandidate> beam;
        beam.push(BeamCandidate(prompt, 0.0f));
        
        for (int step = 0; step < max_length; ++step) {
            std::priority_queue<BeamCandidate> next_beam;
            
            // Expand each candidate in current beam
            while (!beam.empty()) {
                auto candidate = beam.top();
                beam.pop();
                
                auto input_tensor = torch::tensor({candidate.sequence}, torch::kLong).to(device);
                auto logits = model.forward(input_tensor);
                auto log_probs = torch::log_softmax(logits, -1);
                
                // Get top k tokens for this candidate
                auto [top_probs, top_indices] = torch::topk(log_probs, beam_width);
                
                for (int k = 0; k < beam_width; ++k) {
                    auto new_sequence = candidate.sequence;
                    int64_t token_id = top_indices[0][k].item<int64_t>();
                    float token_prob = top_probs[0][k].item<float>();
                    
                    new_sequence.push_back(token_id);
                    float new_score = candidate.score + token_prob;
                    
                    next_beam.push(BeamCandidate(new_sequence, new_score));
                }
            }
            
            // Keep only top beam_width candidates
            beam = std::priority_queue<BeamCandidate>();
            for (int i = 0; i < beam_width && !next_beam.empty(); ++i) {
                beam.push(next_beam.top());
                next_beam.pop();
            }
        }
        
        // Return best sequence
        return beam.top().sequence;
    }
    
    // 3. Top-k Sampling - Sample from top k most likely tokens
    std::vector<int64_t> top_k_sampling(std::vector<int64_t> prompt, int k = 10, 
                                       float temperature = 1.0, int max_length = 50) {
        torch::NoGradGuard no_grad;
        model.eval();
        
        std::vector<int64_t> sequence = prompt;
        
        for (int i = 0; i < max_length; ++i) {
            auto input_tensor = torch::tensor({sequence}, torch::kLong).to(device);
            auto logits = model.forward(input_tensor) / temperature;
            
            // Get top k tokens
            auto [top_logits, top_indices] = torch::topk(logits, k);
            
            // Convert to probabilities and sample
            auto probs = torch::softmax(top_logits, -1);
            auto sampled_idx = torch::multinomial(probs, 1);
            
            // Map back to original vocabulary
            int64_t token_id = top_indices[0][sampled_idx[0][0]].item<int64_t>();
            sequence.push_back(token_id);
        }
        
        return sequence;
    }
    
    // 4. Top-p (Nucleus) Sampling - Sample from smallest set with cumulative prob >= p
    std::vector<int64_t> top_p_sampling(std::vector<int64_t> prompt, float p = 0.9, 
                                       float temperature = 1.0, int max_length = 50) {
        torch::NoGradGuard no_grad;
        model.eval();
        
        std::vector<int64_t> sequence = prompt;
        
        for (int i = 0; i < max_length; ++i) {
            auto input_tensor = torch::tensor({sequence}, torch::kLong).to(device);
            auto logits = model.forward(input_tensor) / temperature;
            
            // Sort probabilities in descending order
            auto probs = torch::softmax(logits, -1);
            auto [sorted_probs, sorted_indices] = torch::sort(probs, -1, true);
            
            // Calculate cumulative probabilities
            auto cumsum_probs = torch::cumsum(sorted_probs, -1);
            
            // Find cutoff where cumsum >= p
            auto mask = cumsum_probs <= p;
            
            // Ensure at least one token is kept
            mask[0][0] = true;
            
            // Zero out probabilities beyond cutoff
            auto filtered_probs = sorted_probs * mask.to(torch::kFloat);
            
            // Renormalize and sample
            filtered_probs = filtered_probs / torch::sum(filtered_probs, -1, true);
            auto sampled_idx = torch::multinomial(filtered_probs, 1);
            
            // Map back to original vocabulary
            int64_t token_id = sorted_indices[0][sampled_idx[0][0]].item<int64_t>();
            sequence.push_back(token_id);
        }
        
        return sequence;
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
    
    // Create model
    int vocab_size = 1000;
    int embed_dim = 256;
    SimpleLLM model(vocab_size, embed_dim);
    model.to(device);
    
    std::cout << "=== Text Generation Sampling Methods ===" << std::endl;
    std::cout << "Vocabulary size: " << vocab_size << std::endl;
    
    // Initialize generator
    TextGenerator generator(model, device);
    
    // Test prompt
    std::vector<int64_t> prompt = {1, 42, 123}; // Example tokens
    
    std::cout << "\nPrompt tokens: ";
    for (auto token : prompt) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    // 1. Greedy Search
    std::cout << "\n=== Greedy Search ===" << std::endl;
    auto greedy_result = generator.greedy_search(prompt, 10);
    std::cout << "Generated: ";
    for (size_t i = prompt.size(); i < greedy_result.size(); ++i) {
        std::cout << greedy_result[i] << " ";
    }
    std::cout << std::endl;
    
    // 2. Beam Search
    std::cout << "\n=== Beam Search (width=3) ===" << std::endl;
    auto beam_result = generator.beam_search(prompt, 3, 10);
    std::cout << "Generated: ";
    for (size_t i = prompt.size(); i < beam_result.size(); ++i) {
        std::cout << beam_result[i] << " ";
    }
    std::cout << std::endl;
    
    // 3. Top-k Sampling
    std::cout << "\n=== Top-k Sampling (k=10, T=0.8) ===" << std::endl;
    auto topk_result = generator.top_k_sampling(prompt, 10, 0.8, 10);
    std::cout << "Generated: ";
    for (size_t i = prompt.size(); i < topk_result.size(); ++i) {
        std::cout << topk_result[i] << " ";
    }
    std::cout << std::endl;
    
    // 4. Top-p Sampling
    std::cout << "\n=== Top-p Sampling (p=0.9, T=0.8) ===" << std::endl;
    auto topp_result = generator.top_p_sampling(prompt, 0.9, 0.8, 10);
    std::cout << "Generated: ";
    for (size_t i = prompt.size(); i < topp_result.size(); ++i) {
        std::cout << topp_result[i] << " ";
    }
    std::cout << std::endl;
    
    // Compare multiple runs of stochastic methods
    std::cout << "\n=== Diversity Comparison (3 runs each) ===" << std::endl;
    
    std::cout << "\nTop-k Sampling variations:" << std::endl;
    for (int run = 0; run < 3; ++run) {
        auto result = generator.top_k_sampling(prompt, 10, 1.0, 5);
        std::cout << "Run " << run + 1 << ": ";
        for (size_t i = prompt.size(); i < result.size(); ++i) {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nTop-p Sampling variations:" << std::endl;
    for (int run = 0; run < 3; ++run) {
        auto result = generator.top_p_sampling(prompt, 0.9, 1.0, 5);
        std::cout << "Run " << run + 1 << ": ";
        for (size_t i = prompt.size(); i < result.size(); ++i) {
            std::cout << result[i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n=== Method Characteristics ===" << std::endl;
    std::cout << "Greedy: Deterministic, always picks most likely token" << std::endl;
    std::cout << "Beam Search: Explores multiple paths, finds high-probability sequences" << std::endl;
    std::cout << "Top-k: Samples from k most likely tokens, controlled randomness" << std::endl;
    std::cout << "Top-p: Dynamic vocabulary size, maintains probability mass" << std::endl;
    
    return 0;
}