// Transformer model combining encoder and decoder
#include <torch/torch.h>
#include <iostream>
#include "positional_encoding.h"
#include "encoder.h"
#include "decoder.h"

struct Transformer : torch::nn::Module {
    Encoder encoder{nullptr};
    Decoder decoder{nullptr};
    
    Transformer(int num_layers, int d_model, int nhead, int d_ff) {
        encoder = register_module("encoder", Encoder(num_layers, d_model, nhead, d_ff));
        decoder = register_module("decoder", Decoder(num_layers, d_model, nhead, d_ff));
    }
    
    torch::Tensor forward(torch::Tensor src, torch::Tensor tgt, torch::Tensor tgt_mask) {
        auto enc_out = encoder->forward(src);
        return decoder->forward(tgt, enc_out, tgt_mask);
    }
};

int main() {
    int src_len = 5, tgt_len = 4, d_model = 512, nhead = 8, d_ff = 2048, num_layers = 6;
    
    Transformer transformer(num_layers, d_model, nhead, d_ff);
    transformer->eval();
    
    // Create input tensors
    auto src = torch::ones({src_len, 1, d_model});
    auto tgt = torch::ones({tgt_len, 1, d_model});
    
    // Add positional encoding
    auto src_pe = positional_encoding(src_len, d_model).unsqueeze(1);
    auto tgt_pe = positional_encoding(tgt_len, d_model).unsqueeze(1);
    src = src + src_pe;
    tgt = tgt + tgt_pe;
    
    // Create causal mask for decoder
    auto tgt_mask = torch::triu(torch::ones({tgt_len, tgt_len}) * -1e9, 1);
    auto output = transformer->forward(src, tgt, tgt_mask);
    
    std::cout << "Transformer output shape: " << output.sizes() << std::endl;
    
    return 0;
}
