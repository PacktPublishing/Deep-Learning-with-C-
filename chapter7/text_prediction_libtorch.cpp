#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>

class TextPredictorTorch {
private:
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear output_layer{nullptr};
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;
    int vocab_size = 0;
    int hidden_size;
    torch::Device device;
    
public:
    TextPredictorTorch(int hidden_sz = 128) 
        : hidden_size(hidden_sz), device(torch::kCPU) {
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "Using CUDA" << std::endl;
        }
    }
    
    void buildVocabulary(const std::vector<std::string>& corpus, int min_freq = 2) {
        std::unordered_map<std::string, int> word_counts;
        
        for (const auto& text : corpus) {
            std::istringstream iss(text);
            std::string word;
            while (iss >> word) {
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                word_counts[word]++;
            }
        }
        
        word_to_id["<START>"] = vocab_size++;
        word_to_id["<END>"] = vocab_size++;
        word_to_id["<UNK>"] = vocab_size++;
        
        for (const auto& pair : word_counts) {
            if (pair.second >= min_freq) {
                word_to_id[pair.first] = vocab_size++;
            }
        }
        
        for (const auto& pair : word_to_id) {
            id_to_word[pair.second] = pair.first;
        }
        
        lstm = torch::nn::LSTM(torch::nn::LSTMOptions(vocab_size, hidden_size).batch_first(true));
        output_layer = torch::nn::Linear(hidden_size, vocab_size);
        
        lstm->to(device);
        output_layer->to(device);
        
        std::cout << "Built vocabulary with " << vocab_size << " words" << std::endl;
    }
    
    std::vector<int> textToSequence(const std::string& text) {
        std::vector<int> sequence;
        std::istringstream iss(text);
        std::string word;
        
        sequence.push_back(word_to_id["<START>"]);
        
        while (iss >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            auto it = word_to_id.find(word);
            sequence.push_back(it != word_to_id.end() ? it->second : word_to_id["<UNK>"]);
        }
        
        sequence.push_back(word_to_id["<END>"]);
        return sequence;
    }
    
    std::string sequenceToText(const std::vector<int>& sequence) {
        std::string text;
        for (int id : sequence) {
            if (id == word_to_id["<START>"] || id == word_to_id["<END>"]) continue;
            auto it = id_to_word.find(id);
            if (it != id_to_word.end()) {
                if (!text.empty()) text += " ";
                text += it->second;
            }
        }
        return text;
    }
    
    void train(const std::vector<std::string>& corpus, int epochs = 100, float learning_rate = 0.001f) {
        torch::optim::Adam optimizer(torch::nn::ModuleList({lstm, output_layer})->parameters(), learning_rate);
        
        std::vector<std::vector<int>> sequences;
        for (const auto& text : corpus) {
            auto seq = textToSequence(text);
            if (seq.size() > 2) sequences.push_back(seq);
        }
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            
            for (const auto& sequence : sequences) {
                if (sequence.size() < 2) continue;
                
                auto input_ids = torch::tensor(std::vector<int>(sequence.begin(), sequence.end() - 1)).unsqueeze(0).to(device);
                auto target_ids = torch::tensor(std::vector<int>(sequence.begin() + 1, sequence.end())).to(device);
                
                optimizer.zero_grad();
                
                auto lstm_out = lstm->forward(torch::nn::functional::one_hot(input_ids, vocab_size).to(torch::kFloat));
                auto logits = output_layer->forward(std::get<0>(lstm_out).squeeze(0));
                
                auto loss = torch::nn::functional::cross_entropy(logits, target_ids);
                loss.backward();
                optimizer.step();
                
                total_loss += loss.item<float>();
            }
            
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << total_loss / sequences.size() << std::endl;
            }
        }
    }
    
    std::string predictNextWord(const std::string& context) {
        auto sequence = textToSequence(context);
        auto input_tensor = torch::nn::functional::one_hot(
            torch::tensor(sequence).unsqueeze(0).to(device), vocab_size).to(torch::kFloat);
        
        torch::NoGradGuard no_grad;
        auto lstm_out = lstm->forward(input_tensor);
        auto logits = output_layer->forward(std::get<0>(lstm_out)[0][-1]);
        auto probabilities = torch::softmax(logits, 0);
        
        auto best_id = torch::argmax(probabilities).item<int>();
        auto it = id_to_word.find(best_id);
        return (it != id_to_word.end()) ? it->second : "<UNK>";
    }
    
    std::string generateText(const std::string& seed, int max_length = 20) {
        auto sequence = textToSequence(seed);
        
        torch::NoGradGuard no_grad;
        
        for (int i = 0; i < max_length; ++i) {
            auto input_tensor = torch::nn::functional::one_hot(
                torch::tensor(sequence).unsqueeze(0).to(device), vocab_size).to(torch::kFloat);
            
            auto lstm_out = lstm->forward(input_tensor);
            auto logits = output_layer->forward(std::get<0>(lstm_out)[0][-1]);
            auto probabilities = torch::softmax(logits, 0);
            
            auto next_id = torch::multinomial(probabilities, 1).item<int>();
            sequence.push_back(next_id);
            
            if (next_id == word_to_id["<END>"]) break;
        }
        
        return sequenceToText(sequence);
    }
    
    void saveModel(const std::string& filename) {
        torch::serialize::OutputArchive archive;
        lstm->save(archive);
        output_layer->save(archive);
        archive.save_to(filename);
        std::cout << "Model saved to " << filename << std::endl;
    }
    
    void loadModel(const std::string& filename) {
        torch::serialize::InputArchive archive;
        archive.load_from(filename);
        lstm->load(archive);
        output_layer->load(archive);
        std::cout << "Model loaded from " << filename << std::endl;
    }
};

int main() {
    TextPredictorTorch predictor(128);
    
    std::vector<std::string> corpus = {
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "natural language processing enables computers to understand human language",
        "deep learning models can learn complex patterns from data",
        "recurrent neural networks are good for sequential data"
    };
    
    predictor.buildVocabulary(corpus, 1);
    predictor.train(corpus, 50, 0.01f);
    
    std::cout << "\nTesting predictions:" << std::endl;
    std::string next_word = predictor.predictNextWord("the quick brown");
    std::cout << "Next word: " << next_word << std::endl;
    
    std::string generated = predictor.generateText("machine learning", 10);
    std::cout << "Generated: " << generated << std::endl;
    
    predictor.saveModel("text_predictor_torch.pt");
    
    return 0;
}