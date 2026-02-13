#include <torch/torch.h>
#include <iostream>
#include "positional_encoding.h"
#include "encoder.h"

int main() {
    int seq_len = 4, d_model = 512, nhead = 8, d_ff = 2048, num_layers = 6;
    
    Encoder encoder(num_layers, d_model, nhead, d_ff);
    encoder->eval();
    
    auto x = torch::ones({seq_len, 1, d_model});
    auto pe = positional_encoding(seq_len, d_model).unsqueeze(1);
    x = x + pe;
    
    auto output = encoder->forward(x);
    
    std::cout << "Encoder output shape: " << output.sizes() << std::endl;
    
    return 0;
}
