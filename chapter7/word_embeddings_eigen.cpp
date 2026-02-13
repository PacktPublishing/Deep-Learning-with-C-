#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>

class WordEmbeddings {
private:
    Eigen::MatrixXf embeddings;
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;
    int vocab_size = 0;
    int embed_dim;
    
public:
    WordEmbeddings(int embedding_dim = 100) : embed_dim(embedding_dim) {}
    
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
        
        // Initialize embeddings randomly
        embeddings = Eigen::MatrixXf::Random(vocab_size, embed_dim) * 0.1f;
        
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
    
    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    void trainSkipGram(const std::string& filename, int epochs = 100, float learning_rate = 0.01f) {
        auto pairs = generateSkipGramPairs(filename);
        Eigen::MatrixXf output_weights = Eigen::MatrixXf::Random(embed_dim, vocab_size) * 0.1f;
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            
            for (const auto& pair : pairs) {
                int center_word = pair.first;
                int context_word = pair.second;
                
                // Forward pass
                Eigen::VectorXf center_embed = embeddings.row(center_word);
                Eigen::VectorXf scores = output_weights.transpose() * center_embed;
                
                // Softmax
                float max_score = scores.maxCoeff();
                Eigen::VectorXf exp_scores = (scores.array() - max_score).exp();
                float sum_exp = exp_scores.sum();
                Eigen::VectorXf probs = exp_scores / sum_exp;
                
                // Loss
                total_loss -= std::log(probs(context_word) + 1e-15f);
                
                // Backward pass
                Eigen::VectorXf grad_output = probs;
                grad_output(context_word) -= 1.0f;
                
                // Update weights
                Eigen::VectorXf grad_embed = output_weights * grad_output;
                embeddings.row(center_word) -= learning_rate * grad_embed.transpose();
                output_weights -= learning_rate * center_embed * grad_output.transpose();
            }
            
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << total_loss / pairs.size() << std::endl;
            }
        }
    }
    
    Eigen::VectorXf getWordEmbedding(const std::string& word) {
        auto it = word_to_id.find(word);
        int word_id = (it != word_to_id.end()) ? it->second : word_to_id.at("<UNK>");
        return embeddings.row(word_id);
    }
    
    std::vector<std::pair<std::string, float>> findSimilarWords(const std::string& word, int top_k = 5) {
        auto target_embed = getWordEmbedding(word);
        std::vector<std::pair<std::string, float>> similarities;
        
        for (const auto& pair : word_to_id) {
            if (pair.first == word || pair.first == "<UNK>") continue;
            
            auto other_embed = getWordEmbedding(pair.first);
            float similarity = target_embed.dot(other_embed) / 
                              (target_embed.norm() * other_embed.norm());
            similarities.emplace_back(pair.first, similarity);
        }
        
        std::sort(similarities.begin(), similarities.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        similarities.resize(std::min(top_k, (int)similarities.size()));
        return similarities;
    }
    
    void saveEmbeddings(const std::string& filename) {
        std::ofstream file(filename);
        
        for (const auto& pair : word_to_id) {
            auto embed = getWordEmbedding(pair.first);
            file << pair.first;
            for (int i = 0; i < embed_dim; ++i) {
                file << " " << embed(i);
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