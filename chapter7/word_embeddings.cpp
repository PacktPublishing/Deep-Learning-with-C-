#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

class WordEmbeddings {
private:
    torch::nn::Embedding embedding{nullptr};
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;
    int vocab_size = 0;
    int embed_dim;
    torch::Device device;
    
public:
    WordEmbeddings(int embedding_dim = 100) 
        : embed_dim(embedding_dim), device(torch::kCPU) {
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
        }
    }
    
    void buildVocabulary(const std::string& filename, int min_freq = 2) {
        std::ifstream file(filename);
        std::unordered_map<std::string, int> word_counts;
        std::string line, word;
        
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            while (iss >> word) {
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                word.erase(std::remove_if(word.begin(), word.end(), 
                    [](char c) { return !std::isalnum(c); }), word.end());
                if (!word.empty()) word_counts[word]++;
            }
        }
        
        word_to_id["<UNK>"] = vocab_size++;
        
        for (const auto& pair : word_counts) {
            if (pair.second >= min_freq) {
                word_to_id[pair.first] = vocab_size++;
            }
        }
        
        for (const auto& pair : word_to_id) {
            id_to_word[pair.second] = pair.first;
        }
        
        embedding = torch::nn::Embedding(vocab_size, embed_dim);
        embedding->to(device);
        
        std::cout << "Built vocabulary with " << vocab_size << " words" << std::endl;
    }
    
    std::vector<std::pair<int, int>> generateSkipGramPairs(const std::string& filename, int window_size = 2) {
        std::vector<std::pair<int, int>> pairs;
        std::ifstream file(filename);
        std::string line, word;
        
        while (std::getline(file, line)) {
            std::vector<int> sentence;
            std::istringstream iss(line);
            
            while (iss >> word) {
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                word.erase(std::remove_if(word.begin(), word.end(), 
                    [](char c) { return !std::isalnum(c); }), word.end());
                
                if (!word.empty()) {
                    auto it = word_to_id.find(word);
                    sentence.push_back(it != word_to_id.end() ? it->second : word_to_id["<UNK>"]);
                }
            }
            
            for (int i = 0; i < sentence.size(); ++i) {
                for (int j = std::max(0, i - window_size); 
                     j <= std::min((int)sentence.size() - 1, i + window_size); ++j) {
                    if (i != j) {
                        pairs.emplace_back(sentence[i], sentence[j]);
                    }
                }
            }
        }
        
        return pairs;
    }
    
    void trainSkipGram(const std::string& filename, int epochs = 100, float learning_rate = 0.01f) {
        auto pairs = generateSkipGramPairs(filename);
        torch::nn::Linear output_layer(embed_dim, vocab_size);
        output_layer->to(device);
        
        torch::optim::SGD optimizer({embedding->parameters(), output_layer->parameters()}, learning_rate);
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            
            for (const auto& pair : pairs) {
                auto center_word = torch::tensor({pair.first}).to(device);
                auto context_word = torch::tensor({pair.second}).to(device);
                
                optimizer.zero_grad();
                
                auto embed_vec = embedding->forward(center_word);
                auto logits = output_layer->forward(embed_vec);
                auto loss = torch::nn::functional::cross_entropy(logits, context_word);
                
                loss.backward();
                optimizer.step();
                
                total_loss += loss.item<float>();
            }
            
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << total_loss / pairs.size() << std::endl;
            }
        }
    }
    
    torch::Tensor getWordEmbedding(const std::string& word) {
        auto it = word_to_id.find(word);
        int word_id = (it != word_to_id.end()) ? it->second : word_to_id.at("<UNK>");
        
        torch::NoGradGuard no_grad;
        return embedding->forward(torch::tensor({word_id}).to(device)).squeeze(0);
    }
    
    std::vector<std::pair<std::string, float>> findSimilarWords(const std::string& word, int top_k = 5) {
        auto target_embed = getWordEmbedding(word);
        std::vector<std::pair<std::string, float>> similarities;
        
        torch::NoGradGuard no_grad;
        for (const auto& pair : word_to_id) {
            if (pair.first == word || pair.first == "<UNK>") continue;
            
            auto other_embed = getWordEmbedding(pair.first);
            float similarity = torch::cosine_similarity(target_embed.unsqueeze(0), 
                                                       other_embed.unsqueeze(0), 1).item<float>();
            similarities.emplace_back(pair.first, similarity);
        }
        
        std::sort(similarities.begin(), similarities.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        similarities.resize(std::min(top_k, (int)similarities.size()));
        return similarities;
    }
    
    void saveEmbeddings(const std::string& filename) {
        std::ofstream file(filename);
        torch::NoGradGuard no_grad;
        
        for (const auto& pair : word_to_id) {
            auto embed = getWordEmbedding(pair.first);
            file << pair.first;
            for (int i = 0; i < embed_dim; ++i) {
                file << " " << embed[i].item<float>();
            }
            file << "\n";
        }
        
        std::cout << "Embeddings saved to " << filename << std::endl;
    }
};

int main() {
    WordEmbeddings embedder(50);
    
    // Create sample text file
    std::ofstream sample_file("sample_text.txt");
    sample_file << "machine learning is a subset of artificial intelligence\n";
    sample_file << "deep learning models can learn complex patterns from data\n";
    sample_file << "natural language processing enables computers to understand human language\n";
    sample_file << "neural networks are inspired by biological neural networks\n";
    sample_file << "artificial intelligence aims to create intelligent machines\n";
    sample_file.close();
    
    embedder.buildVocabulary("sample_text.txt", 1);
    embedder.trainSkipGram("sample_text.txt", 50, 0.1f);
    
    std::cout << "\nTesting word similarities:" << std::endl;
    auto similar = embedder.findSimilarWords("learning", 3);
    std::cout << "Words similar to 'learning':" << std::endl;
    for (const auto& pair : similar) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
    
    embedder.saveEmbeddings("word_embeddings.txt");
    
    return 0;
}