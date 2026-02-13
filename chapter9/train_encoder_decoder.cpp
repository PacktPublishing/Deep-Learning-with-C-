#include <torch/torch.h>
#include <iostream>
#include "positional_encoding.h"
#include "encoder.h"
#include "decoder.h"

int main() {
    int vocab_size = 50, d_model = 64, nhead = 4, d_ff = 256, num_layers = 2;
    
    // Encoder-Decoder model (Translation)
    torch::nn::Embedding src_embed(vocab_size, d_model);
    torch::nn::Embedding tgt_embed(vocab_size, d_model);
    Encoder encoder(num_layers, d_model, nhead, d_ff);
    Decoder decoder(num_layers, d_model, nhead, d_ff);
    torch::nn::Linear output_head(d_model, vocab_size);
    
    torch::optim::Adam optimizer(
        std::vector<torch::Tensor>{
            src_embed->parameters().begin(), src_embed->parameters().end(),
            tgt_embed->parameters().begin(), tgt_embed->parameters().end(),
            encoder->parameters().begin(), encoder->parameters().end(),
            decoder->parameters().begin(), decoder->parameters().end(),
            output_head->parameters().begin(), output_head->parameters().end()
        }.begin(),
        std::vector<torch::Tensor>{
            src_embed->parameters().begin(), src_embed->parameters().end(),
            tgt_embed->parameters().begin(), tgt_embed->parameters().end(),
            encoder->parameters().begin(), encoder->parameters().end(),
            decoder->parameters().begin(), decoder->parameters().end(),
            output_head->parameters().begin(), output_head->parameters().end()
        }.end(),
        0.001
    );
    
    // Training data: "Hello world" -> "Bonjour monde"
    auto src_ids = torch::tensor({{10, 20, 30}}).transpose(0, 1);      // [3, 1] source
    auto tgt_ids = torch::tensor({{15, 25}}).transpose(0, 1);          // [2, 1] target input
    auto labels = torch::tensor({{25, 35}}).transpose(0, 1);           // [2, 1] target output
    
    // Causal mask for decoder
    auto tgt_mask = torch::triu(torch::ones({2, 2}) * -1e9, 1);
    
    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        optimizer.zero_grad();
        
        // Encode source
        auto src = src_embed->forward(src_ids);
        auto src_pe = positional_encoding(3, d_model).unsqueeze(1);
        src = src + src_pe;
        auto enc_out = encoder->forward(src);
        
        // Decode target
        auto tgt = tgt_embed->forward(tgt_ids);
        auto tgt_pe = positional_encoding(2, d_model).unsqueeze(1);
        tgt = tgt + tgt_pe;
        auto dec_out = decoder->forward(tgt, enc_out, tgt_mask);
        
        // Predict
        auto logits = output_head->forward(dec_out).view({-1, vocab_size});
        auto loss = torch::nn::functional::cross_entropy(logits, labels.view({-1}));
        
        loss.backward();
        optimizer.step();
        
        if (epoch % 20 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }
    
    std::cout << "Encoder-Decoder training complete!\n";
    return 0;
}
