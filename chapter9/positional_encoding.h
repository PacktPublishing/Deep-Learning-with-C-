#pragma once
#include <torch/torch.h>

/**
 * @brief Generate sinusoidal positional encoding
 * @param seq_len Sequence length
 * @param d_model Model dimension
 * @return Positional encoding tensor of shape [seq_len, d_model]
 */
torch::Tensor positional_encoding(int seq_len, int d_model) {
    auto pe = torch::zeros({seq_len, d_model});
    auto position = torch::arange(0, seq_len).unsqueeze(1);
    auto div_term = torch::exp(torch::arange(0, d_model, 2) * (-std::log(10000.0) / d_model));
    
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                   torch::sin(position * div_term));
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                   torch::cos(position * div_term));
    return pe;
}
