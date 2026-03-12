#include "decoder.h"

// Masked self-attention + cross-attention + FFN decoder block
DecoderBlock::DecoderBlock(int d_model, int nhead, int d_ff) {
    self_attn = register_module("self_attn", torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(d_model, nhead)));
    cross_attn = register_module("cross_attn", torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(d_model, nhead)));
    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    norm3 = register_module("norm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    fc1 = register_module("fc1", torch::nn::Linear(d_model, d_ff));
    fc2 = register_module("fc2", torch::nn::Linear(d_ff, d_model));
}

torch::Tensor DecoderBlock::forward(torch::Tensor x, torch::Tensor enc_out, torch::Tensor mask) {
    auto self_attn_out = std::get<0>(self_attn->forward(x, x, x, torch::Tensor(), false, mask));
    x = norm1->forward(x + self_attn_out);  // Masked self-attention + Add & Norm
    
    auto cross_attn_out = std::get<0>(cross_attn->forward(x, enc_out, enc_out));
    x = norm2->forward(x + cross_attn_out);  // Cross-attention + Add & Norm
    
    auto ff_out = fc2->forward(torch::relu(fc1->forward(x)));
    return norm3->forward(x + ff_out);  // FFN + Add & Norm
}

// Stack of decoder blocks
Decoder::Decoder(int num_layers, int d_model, int nhead, int d_ff) {
    layers = register_module("layers", torch::nn::ModuleList());
    for (int i = 0; i < num_layers; ++i)
        layers->push_back(std::make_shared<DecoderBlock>(d_model, nhead, d_ff));
}

torch::Tensor Decoder::forward(torch::Tensor x, torch::Tensor enc_out, torch::Tensor mask) {
    for (auto& layer : *layers)
        x = layer->as<DecoderBlock>()->forward(x, enc_out, mask);
    return x;
}
