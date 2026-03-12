#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>

class Vocabulary {
private:
    std::unordered_map<std::string, int> word_to_id;
    std::vector<std::string> id_to_word;
    int next_id = 0;
    
public:
    int SOS_TOKEN = 0, EOS_TOKEN = 1, UNK_TOKEN = 2;
    
    Vocabulary() {
        add_word("<SOS>");
        add_word("<EOS>");
        add_word("<UNK>");
    }
    
    int add_word(const std::string& word) {
        if (word_to_id.find(word) == word_to_id.end()) {
            word_to_id[word] = next_id;
            id_to_word.push_back(word);
            return next_id++;
        }
        return word_to_id[word];
    }
    
    int get_id(const std::string& word) {
        return word_to_id.count(word) ? word_to_id[word] : UNK_TOKEN;
    }
    
    std::string get_word(int id) {
        return id < id_to_word.size() ? id_to_word[id] : "<UNK>";
    }
    
    int size() const { return next_id; }
};

class Encoder : public torch::nn::Module {
private:
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTM lstm{nullptr};
    int hidden_size;
    
public:
    Encoder(int vocab_size, int embed_size, int hidden_sz)
        : hidden_size(hidden_sz) {
        embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embed_size));
        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(embed_size, hidden_size).batch_first(true)));
    }
    
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input) {
        auto embedded = embedding->forward(input);
        auto [output, hidden] = lstm->forward(embedded);
        return std::make_tuple(std::get<0>(hidden), std::get<1>(hidden));
    }
};

class Decoder : public torch::nn::Module {
private:
    torch::nn::Embedding embedding{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear output_projection{nullptr};
    int hidden_size;
    
public:
    Decoder(int vocab_size, int embed_size, int hidden_sz)
        : hidden_size(hidden_sz) {
        embedding = register_module("embedding", torch::nn::Embedding(vocab_size, embed_size));
        lstm = register_module("lstm", torch::nn::LSTM(
            torch::nn::LSTMOptions(embed_size, hidden_size).batch_first(true)));
        output_projection = register_module("output", torch::nn::Linear(hidden_size, vocab_size));
    }
    
    torch::Tensor forward(torch::Tensor input, torch::Tensor h0, torch::Tensor c0) {
        auto embedded = embedding->forward(input);
        auto [output, _] = lstm->forward(embedded, std::make_tuple(h0, c0));
        return output_projection->forward(output);
    }
};

class Seq2SeqTranslator : public torch::nn::Module {
private:
    std::shared_ptr<Encoder> encoder;
    std::shared_ptr<Decoder> decoder;
    
public:
    Seq2SeqTranslator(int src_vocab_size, int tgt_vocab_size, int embed_size, int hidden_size) {
        encoder = register_module("encoder", std::make_shared<Encoder>(src_vocab_size, embed_size, hidden_size));
        decoder = register_module("decoder", std::make_shared<Decoder>(tgt_vocab_size, embed_size, hidden_size));
    }
    
    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt) {
        auto [h0, c0] = encoder->forward(src);
        return decoder->forward(tgt, h0, c0);
    }
    
    std::vector<int> translate(torch::Tensor src, Vocabulary& tgt_vocab, int max_length = 20) {
        eval();
        torch::NoGradGuard no_grad;
        
        auto [h0, c0] = encoder->forward(src);
        
        std::vector<int> result;
        std::vector<int64_t> input_data = {static_cast<int64_t>(tgt_vocab.SOS_TOKEN)};
        auto input = torch::from_blob(input_data.data(), {1, 1}, torch::kLong).clone().to(src.device());
        
        for (int i = 0; i < max_length; ++i) {
            auto output = decoder->forward(input, h0, c0);
            auto predicted = torch::argmax(output, -1);
            int token = predicted[0][0].item<int>();
            
            if (token == tgt_vocab.EOS_TOKEN) break;
            result.push_back(token);
            
            // Update input for next iteration
            input_data[0] = token;
            input = torch::from_blob(input_data.data(), {1, 1}, torch::kLong).clone().to(src.device());
        }
        
        return result;
    }
};

std::vector<std::string> tokenize(const std::string& sentence) {
    std::vector<std::string> tokens;
    std::istringstream iss(sentence);
    std::string word;
    while (iss >> word) {
        tokens.push_back(word);
    }
    return tokens;
}

torch::Tensor sentence_to_tensor(const std::vector<std::string>& sentence, Vocabulary& vocab) {
    std::vector<int64_t> ids;
    for (const auto& word : sentence) {
        ids.push_back(vocab.get_id(word));
    }
    ids.push_back(vocab.EOS_TOKEN);
    return torch::from_blob(ids.data(), {1, static_cast<long>(ids.size())}, torch::kLong).clone();
}

int main() {
    std::cout << "Testing LSTM Translator..." << std::endl;
    
    torch::manual_seed(42);
    
    // Simple English-French dataset
    std::vector<std::pair<std::string, std::string>> data = {
        {"hello", "bonjour"}, {"goodbye", "au revoir"}, {"thank you", "merci"},
        {"yes", "oui"}, {"no", "non"}, {"please", "s'il vous plait"},
        {"good morning", "bonjour"}, {"good night", "bonne nuit"},
        {"how are you", "comment allez vous"}, {"I am fine", "je vais bien"}
    };
    
    // Build vocabularies
    Vocabulary src_vocab, tgt_vocab;
    for (const auto& [en, fr] : data) {
        for (const auto& word : tokenize(en)) src_vocab.add_word(word);
        for (const auto& word : tokenize(fr)) tgt_vocab.add_word(word);
    }
    
    std::cout << "Source vocabulary size: " << src_vocab.size() << std::endl;
    std::cout << "Target vocabulary size: " << tgt_vocab.size() << std::endl;
    
    // Model parameters
    const int embed_size = 64;
    const int hidden_size = 128;
    const int num_epochs = 1000;
    const float learning_rate = 0.01f;
    
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;
    
    auto model = std::make_shared<Seq2SeqTranslator>(
        src_vocab.size(), tgt_vocab.size(), embed_size, hidden_size);
    model->to(device);
    
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(learning_rate));
    
    std::cout << "Training translator..." << std::endl;
    
    // Training
    model->train();
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;
        
        for (const auto& [en, fr] : data) {
            auto src_tokens = tokenize(en);
            auto tgt_tokens = tokenize(fr);
            
            auto src_tensor = sentence_to_tensor(src_tokens, src_vocab).to(device);
            auto tgt_input = sentence_to_tensor(tgt_tokens, tgt_vocab).to(device);
            
            // Target for loss (shifted by one position)
            std::vector<int64_t> tgt_ids;
            for (const auto& word : tgt_tokens) {
                tgt_ids.push_back(tgt_vocab.get_id(word));
            }
            tgt_ids.push_back(tgt_vocab.EOS_TOKEN);
            auto tgt_output = torch::from_blob(tgt_ids.data(), {1, static_cast<long>(tgt_ids.size())}, torch::kLong).clone().to(device);
            
            optimizer.zero_grad();
            auto predictions = model->forward(src_tensor, tgt_input);
            auto loss = torch::cross_entropy_loss(
                predictions.view({-1, tgt_vocab.size()}), tgt_output.view({-1}));
            loss.backward();
            optimizer.step();
            
            total_loss += loss.item<float>();
        }
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << total_loss / data.size() << std::endl;
        }
    }
    
    // Test translation
    std::cout << "\nTranslations:" << std::endl;
    for (const auto& test_sentence : {"hello", "thank you", "good morning"}) {
        auto tokens = tokenize(test_sentence);
        auto src_tensor = sentence_to_tensor(tokens, src_vocab).to(device);
        auto translation_ids = model->translate(src_tensor, tgt_vocab);
        
        std::cout << test_sentence << " -> ";
        for (int id : translation_ids) {
            std::cout << tgt_vocab.get_word(id) << " ";
        }
        std::cout << std::endl;
    }
    
    std::cout << "LSTM Translator test completed successfully!" << std::endl;
    
    return 0;
}