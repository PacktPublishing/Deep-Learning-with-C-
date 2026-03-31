#include <torch/torch.h>
#include <vector>
#include <cstdint>
#include <limits>
#include <iostream>

struct Codebook {
    std::vector<float> centroids;
    std::vector<uint8_t> indices;

    uint8_t find_nearest_centroid(float value) {
        uint8_t best = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (uint8_t i = 0; i < centroids.size(); ++i) {
            float d = (value - centroids[i]) * (value - centroids[i]);
            if (d < best_dist) { best_dist = d; best = i; }
        }
        return best;
    }

    void quantize(const torch::Tensor& weights, int num_clusters, int iters = 10) {
        auto flat = weights.flatten().to(torch::kFloat32);
        int n = flat.numel();

        auto perm = torch::randperm(n).slice(0, 0, num_clusters);
        auto centers = flat.index_select(0, perm).clone();

        for (int it = 0; it < iters; ++it) {
            auto dists = (flat.unsqueeze(1) - centers.unsqueeze(0)).pow(2);
            auto assignments = dists.argmin(1);
            for (int c = 0; c < num_clusters; ++c) {
                auto mask = assignments.eq(c);
                if (mask.any().item<bool>())
                    centers[c] = flat.index({mask}).mean();
            }
        }

        centroids.assign(centers.data_ptr<float>(), centers.data_ptr<float>() + num_clusters);
        indices.resize(n);
        auto* w = flat.data_ptr<float>();
        for (int i = 0; i < n; ++i)
            indices[i] = find_nearest_centroid(w[i]);
    }

    torch::Tensor dequantize(const std::vector<int64_t>& shape) {
        auto result = torch::empty({(int64_t)indices.size()});
        auto* ptr = result.data_ptr<float>();
        for (size_t i = 0; i < indices.size(); ++i)
            ptr[i] = centroids[indices[i]];
        return result.reshape(shape);
    }
};

int main() {
    torch::manual_seed(42);

    // 4x4 weight matrix with values clustered around -1, 0, and 1
    auto weights = torch::tensor({
        -1.1f, -0.9f, -1.0f, -0.8f,
         0.1f, -0.1f,  0.2f,  0.0f,
         0.9f,  1.1f,  1.0f,  0.8f,
        -0.9f,  0.0f,  1.0f, -1.0f
    }).reshape({4, 4});

    std::cout << "Original weights:\n" << weights << "\n\n";

    Codebook cb;
    int num_clusters = 3;
    cb.quantize(weights, num_clusters);

    std::cout << "Centroids (" << num_clusters << " clusters): ";
    for (float c : cb.centroids) std::cout << c << " ";
    std::cout << "\n\n";

    std::cout << "Indices: ";
    for (uint8_t idx : cb.indices) std::cout << (int)idx << " ";
    std::cout << "\n\n";

    auto reconstructed = cb.dequantize({4, 4});
    std::cout << "Reconstructed weights:\n" << reconstructed << "\n\n";

    auto error = (weights - reconstructed).abs().mean().item<float>();
    std::cout << "Mean absolute error: " << error << "\n";

    std::cout << "Original size:    " << weights.numel() * sizeof(float) << " bytes\n";
    std::cout << "Compressed size:  " << cb.indices.size() * sizeof(uint8_t) + cb.centroids.size() * sizeof(float) << " bytes\n";

    return 0;
}
