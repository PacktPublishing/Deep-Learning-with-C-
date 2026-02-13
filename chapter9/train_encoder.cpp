#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include "positional_encoding.h"
#include "encoder.h"

int main() {
    int vocab_size = 100, max_seq_len = 10, d_model = 128, nhead = 8, d_ff = 512, num_layers = 6;
    
    // BERT model components
    torch::nn::Embedding token_embed(vocab_size, d_model);
    torch::nn::Embedding segment_embed(2, d_model);  // Sentence A/B embeddings
    Encoder encoder(num_layers, d_model, nhead, d_ff);
    torch::nn::Linear mlm_head(d_model, vocab_size);  // Masked Language Modeling
    torch::nn::Linear nsp_head(d_model, 2);           // Next Sentence Prediction
    
    torch::optim::Adam optimizer(
        std::vector<torch::Tensor>{
            token_embed->parameters().begin(), token_embed->parameters().end(),
            segment_embed->parameters().begin(), segment_embed->parameters().end(),
            encoder->parameters().begin(), encoder->parameters().end(),
            mlm_head->parameters().begin(), mlm_head->parameters().end(),
            nsp_head->parameters().begin(), nsp_head->parameters().end()
        }.begin(),
        std::vector<torch::Tensor>{
            token_embed->parameters().begin(), token_embed->parameters().end(),
            segment_embed->parameters().begin(), segment_embed->parameters().end(),
            encoder->parameters().begin(), encoder->parameters().end(),
            mlm_head->parameters().begin(), mlm_head->parameters().end(),
            nsp_head->parameters().begin(), nsp_head->parameters().end()
        }.end(),
        0.0001
    );
    
    // Training data: Two tasks as in BERT paper
    // Task 1: Masked Language Modeling (MLM)
    // Task 2: Next Sentence Prediction (NSP)
    
    // Special tokens: [CLS]=1, [SEP]=2, [MASK]=3, [PAD]=0
    std::vector<std::vector<int>> sentences = {
        // Sentence pair 1 (IsNext=1)
        {1, 10, 20, 3, 40, 2, 15, 25, 35, 2},     // [CLS] I love [MASK] music [SEP] She plays guitar [SEP]
        // Sentence pair 2 (IsNext=1)
        {1, 50, 3, 70, 2, 80, 90, 12, 2, 0},      // [CLS] The [MASK] shines [SEP] It is bright [SEP] [PAD]
        // Sentence pair 3 (NotNext=0)
        {1, 11, 21, 3, 2, 60, 70, 80, 2, 0},      // [CLS] We eat [MASK] [SEP] Cars are fast [SEP] [PAD]
        // Sentence pair 4 (IsNext=1)
        {1, 30, 3, 50, 60, 2, 70, 80, 90, 2}      // [CLS] He [MASK] a book [SEP] Reading is fun [SEP]
    };
    
    std::vector<std::vector<int>> labels_mlm = {
        {1, 10, 20, 30, 40, 2, 15, 25, 35, 2},    // True token at [MASK]: 30
        {1, 50, 60, 70, 2, 80, 90, 12, 2, 0},     // True token at [MASK]: 60
        {1, 11, 21, 31, 2, 60, 70, 80, 2, 0},     // True token at [MASK]: 31
        {1, 30, 40, 50, 60, 2, 70, 80, 90, 2}     // True token at [MASK]: 40
    };
    
    std::vector<std::vector<int>> segment_ids = {
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1},           // Sentence A=0, Sentence B=1
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0, 1, 1, 1, 1, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 1, 1}
    };
    
    std::vector<int> nsp_labels = {1, 1, 0, 1};  // IsNext=1, NotNext=0
    
    // Training loop
    for (int epoch = 0; epoch < 200; ++epoch) {
        float total_mlm_loss = 0.0, total_nsp_loss = 0.0;
        
        for (size_t i = 0; i < sentences.size(); ++i) {
            optimizer.zero_grad();
            
            auto input_ids = torch::tensor(sentences[i]).view({-1, 1});
            auto segment_id = torch::tensor(segment_ids[i]).view({-1, 1});
            auto mlm_label = torch::tensor(labels_mlm[i]).view({-1, 1});
            auto nsp_label = torch::tensor({nsp_labels[i]});
            
            // BERT forward pass
            auto token_emb = token_embed->forward(input_ids);
            auto seg_emb = segment_embed->forward(segment_id);
            auto x = token_emb + seg_emb;
            auto pe = positional_encoding(input_ids.size(0), d_model).unsqueeze(1);
            x = x + pe;
            
            auto enc_out = encoder->forward(x);
            
            // Task 1: Masked Language Modeling (predict all tokens)
            auto mlm_logits = mlm_head->forward(enc_out).view({-1, vocab_size});
            auto mlm_loss = torch::nn::functional::cross_entropy(mlm_logits, mlm_label.view({-1}));
            
            // Task 2: Next Sentence Prediction (use [CLS] token)
            auto cls_output = enc_out[0];  // [CLS] is first token
            auto nsp_logits = nsp_head->forward(cls_output);
            auto nsp_loss = torch::nn::functional::cross_entropy(nsp_logits, nsp_label);
            
            // Combined loss
            auto loss = mlm_loss + nsp_loss;
            loss.backward();
            optimizer.step();
            
            total_mlm_loss += mlm_loss.item<float>();
            total_nsp_loss += nsp_loss.item<float>();
        }
        
        if (epoch % 40 == 0)
            std::cout << "Epoch " << epoch 
                      << ", MLM Loss: " << total_mlm_loss / sentences.size()
                      << ", NSP Loss: " << total_nsp_loss / sentences.size() << std::endl;
    }
    
    std::cout << "\nBERT training complete!\n";
    return 0;
}
