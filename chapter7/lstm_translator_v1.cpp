#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <sstream>

class Seq2SeqTranslator : public torch::nn::Module {
private:
    torch::nn::Embedding src_embedding{nullptr};
    torch::nn::Embedding tgt_embedding{nullptr};
    torch::nn::LSTM encoder{nullptr};
    torch::nn::LSTM decoder{nullptr};
    torch::nn::Linear output_projection{nullptr};
    
    std::unordered_map<std::string, int> src_vocab;
    std::unordered_map<std::string, int> tgt_vocab;
    std::unordered_map<int, std::string> tgt_id_to_word;
    
    int src_vocab_size = 0;
    int tgt_vocab_size = 0;
    int embed_dim;
    int hidden_dim;
    torch::Device device;
    
public:
    Seq2SeqTranslator(int embed_sz = 64, int hidden_sz = 128) 
        : embed_dim(embed_sz), hidden_dim(hidden_sz), device(torch::kCPU) {
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
        }
    }
    
    void buildVocabulary(const std::vector<std::pair<std::string, std::string>>& pairs) {
        src_vocab["<PAD>"] = src_vocab_size++;
        src_vocab["<SOS>"] = src_vocab_size++;
        src_vocab["<EOS>"] = src_vocab_size++;
        src_vocab["<UNK>"] = src_vocab_size++;
        
        tgt_vocab["<PAD>"] = tgt_vocab_size++;
        tgt_vocab["<SOS>"] = tgt_vocab_size++;
        tgt_vocab["<EOS>"] = tgt_vocab_size++;
        tgt_vocab["<UNK>"] = tgt_vocab_size++;
        
        for (const auto& pair : pairs) {
            std::istringstream src_iss(pair.first), tgt_iss(pair.second);
            std::string word;
            
            while (src_iss >> word) {
                if (src_vocab.find(word) == src_vocab.end()) {
                    src_vocab[word] = src_vocab_size++;
                }
            }
            
            while (tgt_iss >> word) {
                if (tgt_vocab.find(word) == tgt_vocab.end()) {
                    tgt_vocab[word] = tgt_vocab_size++;
                }
            }
        }
        
        for (const auto& pair : tgt_vocab) {
            tgt_id_to_word[pair.second] = pair.first;
        }
        
        // Initialize model components
        src_embedding = torch::nn::Embedding(src_vocab_size, embed_dim);
        tgt_embedding = torch::nn::Embedding(tgt_vocab_size, embed_dim);
        encoder = torch::nn::LSTM(torch::nn::LSTMOptions(embed_dim, hidden_dim).batch_first(true));
        decoder = torch::nn::LSTM(torch::nn::LSTMOptions(embed_dim, hidden_dim).batch_first(true));
        output_projection = torch::nn::Linear(hidden_dim, tgt_vocab_size);
        
        register_module("src_embedding", src_embedding);
        register_module("tgt_embedding", tgt_embedding);
        register_module("encoder", encoder);
        register_module("decoder", decoder);
        register_module("output_projection", output_projection);
        
        to(device);
        
        std::cout << "Built vocabularies: src=" << src_vocab_size << ", tgt=" << tgt_vocab_size << std::endl;
    }
    
    std::vector<int> sentenceToIds(const std::string& sentence, const std::unordered_map<std::string, int>& vocab, bool add_eos = true) {
        std::vector<int> ids;
        ids.push_back(vocab.at("<SOS>"));
        
        std::istringstream iss(sentence);
        std::string word;
        while (iss >> word) {
            auto it = vocab.find(word);
            ids.push_back(it != vocab.end() ? it->second : vocab.at("<UNK>"));
        }
        
        if (add_eos) {
            ids.push_back(vocab.at("<EOS>"));
        }
        
        return ids;
    }
    
    std::string idsToSentence(const std::vector<int>& ids) {
        std::string sentence;
        for (int id : ids) {
            if (id == tgt_vocab.at("<SOS>") || id == tgt_vocab.at("<PAD>")) continue;
            if (id == tgt_vocab.at("<EOS>")) break;
            
            auto it = tgt_id_to_word.find(id);
            if (it != tgt_id_to_word.end()) {
                if (!sentence.empty()) sentence += " ";
                sentence += it->second;
            }
        }
        return sentence;
    }
    
    torch::Tensor forward(torch::Tensor src_ids, torch::Tensor tgt_ids) {
        // Encoder
        auto src_embeds = src_embedding->forward(src_ids);
        auto encoder_out = encoder->forward(src_embeds);
        auto context = std::get<1>(encoder_out);
        
        // Decoder
        auto tgt_embeds = tgt_embedding->forward(tgt_ids);
        auto decoder_out = decoder->forward(tgt_embeds, context);
        auto hidden_states = std::get<0>(decoder_out);
        
        // Output projection
        return output_projection->forward(hidden_states);
    }
    
    void train(const std::vector<std::pair<std::string, std::string>>& pairs, int epochs = 100, float lr = 0.001f) {
        torch::optim::Adam optimizer(parameters(), lr);
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;
            
            for (const auto& pair : pairs) {
                auto src_ids = sentenceToIds(pair.first, src_vocab, false);
                auto tgt_ids = sentenceToIds(pair.second, tgt_vocab, true);
                
                auto src_tensor = torch::tensor(src_ids).unsqueeze(0).to(device);
                auto tgt_input = torch::tensor(std::vector<int>(tgt_ids.begin(), tgt_ids.end() - 1)).unsqueeze(0).to(device);
                auto tgt_output = torch::tensor(std::vector<int>(tgt_ids.begin() + 1, tgt_ids.end())).to(device);
                
                optimizer.zero_grad();
                
                auto logits = forward(src_tensor, tgt_input);
                auto loss = torch::nn::functional::cross_entropy(logits.squeeze(0), tgt_output);
                
                loss.backward();
                optimizer.step();
                
                total_loss += loss.item<float>();
            }
            
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Loss: " << total_loss / pairs.size() << std::endl;
            }
        }
    }
    
    std::string translate(const std::string& source, int max_length = 20) {
        torch::NoGradGuard no_grad;
        
        auto src_ids = sentenceToIds(source, src_vocab, false);
        auto src_tensor = torch::tensor(src_ids).unsqueeze(0).to(device);
        
        // Encode
        auto src_embeds = src_embedding->forward(src_tensor);
        auto encoder_out = encoder->forward(src_embeds);
        auto context = std::get<1>(encoder_out);
        
        // Decode
        std::vector<int> output_ids;
        int current_id = tgt_vocab.at("<SOS>");
        
        for (int i = 0; i < max_length; ++i) {
            auto input_tensor = torch::tensor({current_id}).unsqueeze(0).to(device);
            auto input_embed = tgt_embedding->forward(input_tensor);
            
            auto decoder_out = decoder->forward(input_embed, context);
            auto hidden = std::get<0>(decoder_out);
            context = std::get<1>(decoder_out);
            
            auto logits = output_projection->forward(hidden.squeeze(0));
            current_id = torch::argmax(logits, 1).item<int>();
            
            output_ids.push_back(current_id);
            
            if (current_id == tgt_vocab.at("<EOS>")) break;
        }
        
        return idsToSentence(output_ids);
    }
};

int main() {
    Seq2SeqTranslator translator(32, 64);
    
    // Sample English-Spanish translation pairs
    std::vector<std::pair<std::string, std::string>> training_data = {
        {"hello", "hola"},
        {"goodbye", "adios"},
        {"thank you", "gracias"},
        {"good morning", "buenos dias"},
        {"good night", "buenas noches"},
        {"how are you", "como estas"},
        {"I am fine", "estoy bien"},
        {"what is your name", "como te llamas"},
        {"my name is", "me llamo"},
        {"nice to meet you", "mucho gusto"}
    };
    
    translator.buildVocabulary(training_data);
    translator.train(training_data, 200, 0.01f);
    
    std::cout << "\nTesting translations:" << std::endl;
    std::cout << "hello -> " << translator.translate("hello") << std::endl;
    std::cout << "thank you -> " << translator.translate("thank you") << std::endl;
    std::cout << "good morning -> " << translator.translate("good morning") << std::endl;
    
    return 0;
}