#pragma once
#include <torch/torch.h>

/**
 * @brief Single encoder layer with self-attention and feedforward network
 */
struct EncoderBlock : torch::nn::Module {
    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    
    EncoderBlock(int d_model, int nhead, int d_ff);
    torch::Tensor forward(torch::Tensor x);
};

/**
 * @brief Transformer encoder with stacked encoder layers
 */
struct Encoder : torch::nn::Module {
    torch::nn::ModuleList layers{nullptr};
    
    Encoder(int num_layers, int d_model, int nhead, int d_ff);
    torch::Tensor forward(torch::Tensor x);
};
