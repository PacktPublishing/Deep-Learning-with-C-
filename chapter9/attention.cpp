#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

class Attention {
public:
    // Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    static std::vector<std::vector<double>> scaledDotProductAttention(
        const std::vector<std::vector<double>>& Q,  // queries
        const std::vector<std::vector<double>>& K,  // keys  
        const std::vector<std::vector<double>>& V   // values
    ) {
        int seq_len = Q.size();
        int d_k = K[0].size();
        double scale = 1.0 / std::sqrt(d_k);
        
        // Compute attention scores: QK^T
        auto scores = matmul(Q, transpose(K));
        
        // Scale and apply softmax
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                scores[i][j] *= scale;
            }
            softmax(scores[i]);
        }
        
        // Apply attention to values: scores * V
        return matmul(scores, V);
    }

private:
    static std::vector<std::vector<double>> matmul(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B
    ) {
        int rows = A.size();
        int cols = B[0].size();
        int inner = A[0].size();
        
        std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                for (int k = 0; k < inner; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }
    
    static std::vector<std::vector<double>> transpose(
        const std::vector<std::vector<double>>& matrix
    ) {
        int rows = matrix.size();
        int cols = matrix[0].size();
        std::vector<std::vector<double>> result(cols, std::vector<double>(rows));
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = matrix[i][j];
            }
        }
        return result;
    }
    
    static void softmax(std::vector<double>& vec) {
        double max_val = *std::max_element(vec.begin(), vec.end());
        double sum = 0.0;
        
        for (double& val : vec) {
            val = std::exp(val - max_val);
            sum += val;
        }
        
        for (double& val : vec) {
            val /= sum;
        }
    }
};

int main() {
    // Example usage
    std::vector<std::vector<double>> Q = {{1.0, 0.5}, {0.8, 1.2}};
    std::vector<std::vector<double>> K = {{0.9, 0.7}, {1.1, 0.6}};
    std::vector<std::vector<double>> V = {{2.0, 1.5}, {1.8, 2.2}};
    
    auto result = Attention::scaledDotProductAttention(Q, K, V);
    
    std::cout << "Attention output:" << std::endl;
    for (const auto& row : result) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}