#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <fstream>

// ============================================================================
// Data Loader: Handle dataset loading and batch processing
// ============================================================================

// Simple dataset class
class SimpleDataset {
private:
    std::vector<torch::Tensor> data;
    std::vector<torch::Tensor> labels;
    
public:
    SimpleDataset(std::vector<torch::Tensor> d, std::vector<torch::Tensor> l)
        : data(d), labels(l) {}
    
    size_t size() const { return data.size(); }
    
    std::pair<torch::Tensor, torch::Tensor> get(size_t idx) {
        return {data[idx], labels[idx]};
    }
};

// Batch loader
class DataLoader {
private:
    SimpleDataset& dataset;
    size_t batch_size;
    bool shuffle;
    std::vector<size_t> indices;
    
public:
    DataLoader(SimpleDataset& ds, size_t bs, bool shuf = true)
        : dataset(ds), batch_size(bs), shuffle(shuf) {
        reset();
    }
    
    void reset() {
        indices.clear();
        for (size_t i = 0; i < dataset.size(); ++i) {
            indices.push_back(i);
        }
        
        if (shuffle) {
            std::random_shuffle(indices.begin(), indices.end());
        }
    }
    
    std::vector<std::pair<torch::Tensor, torch::Tensor>> get_batches() {
        std::vector<std::pair<torch::Tensor, torch::Tensor>> batches;
        
        for (size_t i = 0; i < indices.size(); i += batch_size) {
            std::vector<torch::Tensor> batch_data;
            std::vector<torch::Tensor> batch_labels;
            
            size_t end = std::min(i + batch_size, indices.size());
            for (size_t j = i; j < end; ++j) {
                auto [data, label] = dataset.get(indices[j]);
                batch_data.push_back(data);
                batch_labels.push_back(label);
            }
            
            batches.push_back({
                torch::stack(batch_data),
                torch::stack(batch_labels)
            });
        }
        
        return batches;
    }
};

// ============================================================================
// Example Usage
// ============================================================================
int main() {
    // Generate dummy dataset
    std::vector<torch::Tensor> data;
    std::vector<torch::Tensor> labels;
    
    for (int i = 0; i < 100; ++i) {
        data.push_back(torch::randn({784}));
        labels.push_back(torch::tensor(i % 10));
    }
    
    SimpleDataset dataset(data, labels);
    DataLoader loader(dataset, 32, true);
    
    auto batches = loader.get_batches();
    std::cout << "Total batches: " << batches.size() << std::endl;
    
    for (size_t i = 0; i < batches.size(); ++i) {
        auto [batch_data, batch_labels] = batches[i];
        std::cout << "Batch " << i << " - Data: " << batch_data.sizes()
                  << ", Labels: " << batch_labels.sizes() << std::endl;
    }
    
    return 0;
}
