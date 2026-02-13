#include <torch/torch.h>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/FileStore.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>

// ============================================================================
// Transformer Model
// ============================================================================
struct TransformerModel : torch::nn::Module {
    torch::nn::Embedding embedding{nullptr};
    torch::nn::TransformerEncoder encoder{nullptr};
    torch::nn::Linear fc{nullptr};
    
    TransformerModel(int64_t vocab_size, int64_t d_model, 
                     int64_t nhead, int64_t num_layers)
        : vocab_size_(vocab_size), d_model_(d_model) {
        
        embedding = register_module("embedding", 
            torch::nn::Embedding(vocab_size, d_model));
        
        auto encoder_layer = torch::nn::TransformerEncoderLayer(
            torch::nn::TransformerEncoderLayerOptions(d_model, nhead)
                .dim_feedforward(d_model * 4)
                .dropout(0.1));
        
        encoder = register_module("encoder",
            torch::nn::TransformerEncoder(
                torch::nn::TransformerEncoderOptions(encoder_layer, num_layers)));
        
        fc = register_module("fc", torch::nn::Linear(d_model, vocab_size));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = embedding->forward(x) * std::sqrt(d_model_);
        x = encoder->forward(x);
        x = fc->forward(x);
        return x;
    }
    
private:
    int64_t vocab_size_;
    int64_t d_model_;
};

// ============================================================================
// Distributed Sampler
// ============================================================================
class DistributedSampler {
private:
    size_t dataset_size;
    int rank;
    int world_size;
    std::vector<size_t> indices;
    
public:
    DistributedSampler(size_t size, int r, int ws)
        : dataset_size(size), rank(r), world_size(ws) {
        
        size_t per_rank = (dataset_size + world_size - 1) / world_size;
        for (size_t i = rank; i < dataset_size; i += world_size) {
            if (indices.size() < per_rank) {
                indices.push_back(i);
            }
        }
    }
    
    const std::vector<size_t>& get_indices() const { return indices; }
};

// ============================================================================
// DDP Trainer
// ============================================================================
class DDPTrainer {
private:
    std::shared_ptr<torch::nn::Module> model;
    std::shared_ptr<c10d::ProcessGroup> process_group;
    int rank;
    int world_size;
    
public:
    DDPTrainer(std::shared_ptr<torch::nn::Module> m, int r, int ws, 
               const std::string& master_addr)
        : model(m), rank(r), world_size(ws) {
        
        auto store = std::make_shared<c10d::FileStore>("/tmp/ddp_store", ws);
        c10d::ProcessGroupNCCL::Options options;
        process_group = std::make_shared<c10d::ProcessGroupNCCL>(
            store, r, ws, options);
        
        broadcast_parameters();
    }
    
    void broadcast_parameters() {
        for (auto& param : model->parameters()) {
            std::vector<torch::Tensor> tensors = {param.data()};
            c10d::BroadcastOptions opts;
            opts.rootRank = 0;
            process_group->broadcast(tensors, opts)->wait();
        }
    }
    
    void all_reduce_gradients() {
        for (auto& param : model->parameters()) {
            if (param.grad().defined()) {
                std::vector<torch::Tensor> tensors = {param.grad()};
                c10d::AllreduceOptions opts;
                opts.reduceOp = c10d::ReduceOp::SUM;
                process_group->allreduce(tensors, opts)->wait();
                param.grad().div_(world_size);
            }
        }
    }
    
    void train_step(torch::Tensor input, torch::Tensor target,
                   torch::optim::Optimizer& optimizer) {
        optimizer.zero_grad();
        
        auto output = model->forward(input);
        auto loss = torch::nn::functional::cross_entropy(
            output.view({-1, output.size(-1)}),
            target.view({-1}));
        
        loss.backward();
        all_reduce_gradients();
        optimizer.step();
        
        if (rank == 0) {
            std::cout << "Loss: " << loss.item<float>() << std::endl;
        }
    }
};

// ============================================================================
// Checkpoint Functions
// ============================================================================
void save_checkpoint(const std::string& path, 
                    std::shared_ptr<torch::nn::Module> model,
                    torch::optim::Optimizer& optimizer,
                    int epoch, int rank) {
    if (rank == 0) {
        torch::save(model, path);
        std::cout << "Checkpoint saved at epoch " << epoch << std::endl;
    }
}

void load_checkpoint(const std::string& path,
                    std::shared_ptr<torch::nn::Module> model,
                    torch::optim::Optimizer& optimizer,
                    int& start_epoch) {
    std::ifstream file(path);
    if (file.good()) {
        torch::load(model, path);
        start_epoch = 0;
        std::cout << "Checkpoint loaded" << std::endl;
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    // Model hyperparameters
    const int64_t vocab_size = 10000;
    const int64_t d_model = 512;
    const int64_t nhead = 8;
    const int64_t num_layers = 6;
    const int64_t seq_length = 128;
    const int64_t batch_size = 32;
    const size_t dataset_size = 1000;
    const int num_epochs = 10;
    
    // Initialize distributed environment
    int rank = (argc > 1) ? std::stoi(argv[1]) : 0;
    int world_size = (argc > 2) ? std::stoi(argv[2]) : 1;
    int local_rank = rank % torch::cuda::device_count();
    
    std::cout << "Rank " << rank << "/" << world_size << std::endl;
    
    // Set device
    torch::Device device(torch::kCUDA, local_rank);
    
    // Create model
    auto model = std::make_shared<TransformerModel>(
        vocab_size, d_model, nhead, num_layers);
    model->to(device);
    
    // Initialize DDP trainer
    DDPTrainer trainer(model, rank, world_size, "localhost:29500");
    
    // Create optimizer
    torch::optim::Adam optimizer(model->parameters(), 
                                 torch::optim::AdamOptions(1e-4));
    
    // Load checkpoint if exists
    int start_epoch = 0;
    load_checkpoint("checkpoint.pt", model, optimizer, start_epoch);
    
    // Create distributed data loader
    DistributedSampler sampler(dataset_size, rank, world_size);
    
    // Generate dummy dataset
    std::vector<std::pair<torch::Tensor, torch::Tensor>> dataset;
    for (size_t i = 0; i < dataset_size; ++i) {
        auto input = torch::randint(0, vocab_size, {batch_size, seq_length});
        auto target = torch::randint(0, vocab_size, {batch_size, seq_length});
        dataset.push_back({input, target});
    }
    
    // Training loop
    model->train();
    for (int epoch = start_epoch; epoch < num_epochs; ++epoch) {
        if (rank == 0) {
            std::cout << "\nEpoch " << epoch + 1 << "/" << num_epochs << std::endl;
        }
        
        for (auto idx : sampler.get_indices()) {
            auto [input, target] = dataset[idx];
            input = input.to(device);
            target = target.to(device);
            
            trainer.train_step(input, target, optimizer);
        }
        
        // Save checkpoint
        save_checkpoint("checkpoint.pt", model, optimizer, epoch, rank);
    }
    
    if (rank == 0) {
        std::cout << "\nTraining complete!" << std::endl;
    }
    
    return 0;
}
