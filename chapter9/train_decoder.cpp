#include <torch/torch.h>
#include <iostream>
#include "positional_encoding.h"
#include "decoder.h"

int main() {
    int vocab_size = 50, seq_len = 5, d_model = 64, nhead = 4, d_ff = 256, num_layers = 2;
    
    // Decoder-only model (GPT-style)
    torch::nn::Embedding embed(vocab_size, d_model);
    Decoder decoder(num_layers, d_model, nhead, d_ff);
    torch::nn::Linear output_head(d_model, vocab_size);
    
    torch::optim::Adam optimizer(
        std::vector<torch::Tensor>{
            embed->parameters().begin(), embed->parameters().end(),
            decoder->parameters().begin(), decoder->parameters().end(),
            output_head->parameters().begin(), output_head->parameters().end()
        }.begin(),
        std::vector<torch::Tensor>{
            embed->parameters().begin(), embed->parameters().end(),
            decoder->parameters().begin(), decoder->parameters().end(),
            output_head->parameters().begin(), output_head->parameters().end()
        }.end(),
        0.001
    );
    
    // Training data: "I love AI" -> predict next token
    auto input_ids = torch::tensor({{10, 20, 25}}).transpose(0, 1);    // [3, 1]
    auto labels = torch::tensor({{20, 25, 30}}).transpose(0, 1);       // Shifted by 1
    
    // Causal mask
    auto mask = torch::triu(torch::ones({3, 3}) * -1e9, 1);
    
    // Training loop
    for (int epoch = 0; epoch < 100; ++epoch) {
        optimizer.zero_grad();
        
        auto x = embed->forward(input_ids);
        auto pe = positional_encoding(3, d_model).unsqueeze(1);
        x = x + pe;
        
        // Decoder-only: use x as both input and "encoder output" for cross-attention
        auto enc_out = x.detach();  // Dummy encoder output
        x = decoder->forward(x, enc_out, mask);
        auto logits = output_head->forward(x).view({-1, vocab_size});
        
        auto loss = torch::nn::functional::cross_entropy(logits, labels.view({-1}));
        loss.backward();
        optimizer.step();
        
        if (epoch % 20 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << loss.item<float>() << std::endl;
    }
    
    std::cout << "Decoder training complete!\n";
    return 0;
}
