#include <iostream>
#include <vector>
#include <cmath>

using Matrix = std::vector<std::vector<float>>;

Matrix positional_encoding(int seq_len, int d_model) {
    Matrix pe(seq_len, std::vector<float>(d_model));
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < d_model; i += 2) {
            float div_term = std::exp(i * -std::log(10000.0f) / d_model);
            pe[pos][i] = std::sin(pos * div_term);
            if (i + 1 < d_model)
                pe[pos][i + 1] = std::cos(pos * div_term);
        }
    }
    return pe;
}

int main() {
    int seq_len = 10, d_model = 8;
    Matrix pe = positional_encoding(seq_len, d_model);
    
    std::cout << "Positional Encoding:\n";
    for (const auto& row : pe) {
        for (float val : row)
            std::cout << val << " ";
        std::cout << "\n";
    }
    
    return 0;
}
