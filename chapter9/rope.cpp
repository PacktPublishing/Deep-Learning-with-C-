#include <iostream>
#include <vector>
#include <cmath>

void apply_rope(std::vector<std::vector<float>>& q, std::vector<std::vector<float>>& k, int seq_len, int d) {
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < d; i += 2) {
            float theta = pos / std::pow(10000.0f, (float)i / d);
            float cos_theta = std::cos(theta);
            float sin_theta = std::sin(theta);
            
            // Rotate query
            float q0 = q[pos][i];
            float q1 = q[pos][i + 1];
            q[pos][i] = q0 * cos_theta - q1 * sin_theta;
            q[pos][i + 1] = q0 * sin_theta + q1 * cos_theta;
            
            // Rotate key
            float k0 = k[pos][i];
            float k1 = k[pos][i + 1];
            k[pos][i] = k0 * cos_theta - k1 * sin_theta;
            k[pos][i + 1] = k0 * sin_theta + k1 * cos_theta;
        }
    }
}

int main() {
    int seq_len = 4, d = 8;
    std::vector<std::vector<float>> q(seq_len, std::vector<float>(d, 1.0f));
    std::vector<std::vector<float>> k(seq_len, std::vector<float>(d, 1.0f));
    
    apply_rope(q, k, seq_len, d);
    
    std::cout << "Query after RoPE:\n";
    for (const auto& row : q) {
        for (float val : row) std::cout << val << " ";
        std::cout << "\n";
    }
    
    return 0;
}
