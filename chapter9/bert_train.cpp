#include <torch/torch.h>
#include <iostream>
#include "positional_encoding.h"
#include "encoder.h"

struct BERT : torch::nn::Module {
    torch::nn::Embedding token_embed{nullptr};
    Encoder encoder{nullptr};
    torch::nn::Linear mlm_head{nullptr};
    
    BERT(int vocab_size, int d_model, int nhead, int d_ff, int num_layers) {
        token_embed = register_module("token_embed", torch::nn::Embedding(vocab_size, d_model));
        encoder = register_module("encoder", Encoder(num_layers, d_model, nhead, d_ff));
        mlm_head = register_module("mlm_head", torch::nn::Linear(d_model, vocab_size));
    }
    
    torch::Tensor forward(torch::Tensor input_ids) {
        auto x = token_embed->forward(input_ids);
        auto pe = positional_encoding(input_ids.size(0), x.size(2)).unsqueeze(1);
        x = x + pe;
        x = encoder->forward(x);
        return mlm_head->forward(x);
    }
};

int main() {
    int vocab_size = 100, seq_len = 8, d_model = 128, nhead = 4, d_ff = 512, num_layers = 2;
    
    BERT bert(vocab_size, d_model, nhead, d_ff, num_layers);
    
    // Training data: Multiple sentences with masked tokens
    // Vocabulary: [PAD]=0, [CLS]=1, [SEP]=2, [MASK]=50, tokens=3-49
    std::vector<std::vector<int>> sentences = {
        {1, 10, 20, 50, 30, 2},      // [CLS] I love [MASK] music [SEP] -> AI
        {1, 15, 50, 25, 35, 2},      // [CLS] The [MASK] is blue [SEP] -> sky
        {1, 40, 50, 45, 12, 2},      // [CLS] She [MASK] a book [SEP] -> reads
        {1, 8, 22, 50, 18, 2}        // [CLS] We eat [MASK] daily [SEP] -> food
    };
    
    std::vector<std::vector<int>> labels = {
        {1, 10, 20, 30, 30, 2},      // True token at position 3: 30
        {1, 15, 24, 25, 35, 2},      // True token at position 2: 24
        {1, 40, 42, 45, 12, 2},      // True token at position 2: 42
        {1, 8, 22, 28, 18, 2}        // True token at position 3: 28
    };
    
    torch::optim::Adam optimizer(bert.parameters(), 0.001);
    
    // Training loop
    for (int epoch = 0; epoch < 50; ++epoch) {
        float total_loss = 0.0;
        
        for (size_t i = 0; i < sentences.size(); ++i) {
            optimizer.zero_grad();
            
            auto input_ids = torch::tensor(sentences[i]).view({-1, 1});  // [seq_len, 1]
            auto target_ids = torch::tensor(labels[i]).view({-1, 1});
            
            auto logits = bert.forward(input_ids);      // [seq_len, 1, vocab_size]
            logits = logits.view({-1, vocab_size});     // [seq_len, vocab_size]
            auto targets = target_ids.view({-1});       // [seq_len]
            
            auto loss = torch::nn::functional::cross_entropy(logits, targets);
            loss.backward();
            optimizer.step();
            
            total_loss += loss.item<float>();
        }
        
        if (epoch % 10 == 0)
            std::cout << "Epoch " << epoch << ", Avg Loss: " << total_loss / sentences.size() << std::endl;
    }
    
    // Test predictions
    bert.eval();
    std::cout << "\nPredictions:\n";
    for (size_t i = 0; i < sentences.size(); ++i) {
        auto input_ids = torch::tensor(sentences[i]).view({-1, 1});
        auto output = bert.forward(input_ids);
        
        // Find [MASK] position and predict
        for (size_t j = 0; j < sentences[i].size(); ++j) {
            if (sentences[i][j] == 50) {  // [MASK] token
                auto predicted = output[j][0].argmax().item<int>();
                std::cout << "Sentence " << i << ": Predicted=" << predicted 
                          << ", True=" << labels[i][j] << std::endl;
            }
        }
    }
    
    return 0;
}
