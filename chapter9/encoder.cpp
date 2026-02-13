#include "encoder.h"

// Self-attention + FFN encoder block
EncoderBlock::EncoderBlock(int d_model, int nhead, int d_ff) {
    attn = register_module("attn", torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(d_model, nhead)));
    norm1 = register_module("norm1", torch::nn::LayerNorm(d_model));
    norm2 = register_module("norm2", torch::nn::LayerNorm(d_model));
    fc1 = register_module("fc1", torch::nn::Linear(d_model, d_ff));
    fc2 = register_module("fc2", torch::nn::Linear(d_ff, d_model));
}

torch::Tensor EncoderBlock::forward(torch::Tensor x) {
    auto attn_out = std::get<0>(attn->forward(x, x, x));
    x = norm1->forward(x + attn_out);  // Add & Norm
    auto ff_out = fc2->forward(torch::relu(fc1->forward(x)));
    return norm2->forward(x + ff_out);  // Add & Norm
}

// Stack of encoder blocks
Encoder::Encoder(int num_layers, int d_model, int nhead, int d_ff) {
    layers = register_module("layers", torch::nn::ModuleList());
    for (int i = 0; i < num_layers; ++i)
        layers->push_back(std::make_shared<EncoderBlock>(d_model, nhead, d_ff));
}

torch::Tensor Encoder::forward(torch::Tensor x) {
    for (auto& layer : *layers)
        x = layer->as<EncoderBlock>()->forward(x);
    return x;
}
