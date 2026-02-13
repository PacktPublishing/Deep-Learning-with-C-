#include <torch/torch.h>
#include <iostream>
#include "positional_encoding.h"
#include "decoder.h"

int main() {
    int seq_len = 4, d_model = 512, nhead = 8, d_ff = 2048, num_layers = 6;
    
    Decoder decoder(num_layers, d_model, nhead, d_ff);
    decoder->eval();
    
    auto tgt = torch::ones({seq_len, 1, d_model});
    auto enc_out = torch::randn({seq_len, 1, d_model});
    auto pe = positional_encoding(seq_len, d_model).unsqueeze(1);
    tgt = tgt + pe;
    
    auto mask = torch::triu(torch::ones({seq_len, seq_len}) * -1e9, 1);
    auto output = decoder->forward(tgt, enc_out, mask);
    
    std::cout << "Decoder output shape: " << output.sizes() << std::endl;
    
    return 0;
}
