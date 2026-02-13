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
// FSDP Trainer
// ============================================================================
class FSDPTrainer {
private:
    std::shared_ptr<torch::nn::Module> model;
    std::shared_ptr<c10d::ProcessGroup> process_group;
    int rank;
    int world_size;
    
    std::vector<torch::Tensor> sharded_params;
    std::vector<torch::Tensor> full_params;
    
public:
    FSDPTrainer(std::shared_ptr<torch::nn::Module> m, int r, int ws, 
                const std::string& master_addr)
        : model(m), rank(r), world_size(ws) {
        
        auto store = std::make_shared<c10d::FileStore>("/tmp/fsdp_store", ws);
        c10d::ProcessGroupNCCL::Options options;
        process_group = std::make_shared<c10d::ProcessGroupNCCL>(
            store, r, ws, options);
        
        broadcast_parameters();
        shard_parameters();
    }
    
    void broadcast_parameters() {
        for (auto& param : model->parameters()) {
            std::vector<torch::Tensor> tensors = {param.data()};
            c10d::BroadcastOptions opts;
            opts.rootRank = 0;
            process_group->broadcast(tensors, opts)->wait();
        }
    }
    
    void shard_parameters() {
        std::vector<torch::Tensor> all_params;
        for (const auto& param : model->parameters()) {
            all_params.push_back(param.data().flatten());
        }
        
        torch::Tensor concat_params = torch::cat(all_params);
        int64_t total_size = concat_params.numel();
        
        int64_t shard_size = (total_size + world_size - 1) / world_size;
        int64_t start_idx = rank * shard_size;
        int64_t end_idx = std::min(start_idx + shard_size, total_size);
        
        if (start_idx < total_size) {
            torch::Tensor shard = concat_params.slice(0, start_idx, end_idx).clone();
            sharded_params.push_back(shard);
        }
        
        std::cout << "[Rank " << rank << "] Sharded " << (end_idx - start_idx) 
                  << "/" << total_size << " parameters" << std::endl;
    }
    
    void all_gather_parameters() {
        full_params.clear();
        
        for (const auto& shard : sharded_params) {
            std::vector<torch::Tensor> gathered_shards;
            for (int i = 0; i < world_size; ++i) {
                gathered_shards.push_back(torch::empty_like(shard));
            }
            
            std::vector<std::vector<torch::Tensor>> output_tensors = {gathered_shards};
            std::vector<torch::Tensor> input_tensors = {shard};
            
            c10d::AllgatherOptions opts;
            process_group->allgather(output_tensors, input_tensors, opts)->wait();
            
            torch::Tensor full_param = torch::cat(gathered_shards);
            full_params.push_back(full_param);
        }
        
        update_model_params(full_params);
    }
    
    void update_model_params(const std::vector<torch::Tensor>& params) {
        if (params.empty()) return;
        
        torch::Tensor concat = torch::cat(params);
        int64_t offset = 0;
        
        for (auto& param : model->parameters()) {
            int64_t numel = param.numel();
            torch::Tensor param_data = concat.slice(0, offset, offset + numel)
                                             .view(param.sizes());
            param.data().copy_(param_data);
            offset += numel;
        }
    }
    
    void reduce_scatter_gradients() {
        std::vector<torch::Tensor> all_grads;
        for (const auto& param : model->parameters()) {
            if (param.grad().defined()) {
                all_grads.push_back(param.grad().flatten());
            }
        }
        
        if (all_grads.empty()) return;
        
        torch::Tensor concat_grads = torch::cat(all_grads);
        int64_t total_size = concat_grads.numel();
        int64_t shard_size = (total_size + world_size - 1) / world_size;
        
        std::vector<torch::Tensor> input_list;
        for (int i = 0; i < world_size; ++i) {
            int64_t start = i * shard_size;
            int64_t end = std::min(start + shard_size, total_size);
            if (start < total_size) {
                input_list.push_back(concat_grads.slice(0, start, end));
            }
        }
        
        torch::Tensor output = torch::empty_like(input_list[rank]);
        std::vector<torch::Tensor> output_list = {output};
        
        c10d::ReduceScatterOptions opts;
        opts.reduceOp = c10d::ReduceOp::SUM;
        process_group->reduce_scatter(output_list, {input_list}, opts)->wait();
        
        output.div_(world_size);
    }
    
    void train_step(torch::Tensor input, torch::Tensor target,
                   torch::optim::Optimizer& optimizer) {
        // FSDP: All-gather parameters
        all_gather_parameters();
        
        optimizer.zero_grad();
        
        auto output = model->forward(input);
        auto loss = torch::nn::functional::cross_entropy(
            output.view({-1, output.size(-1)}),
            target.view({-1}));
        
        loss.backward();
        
        // FSDP: Reduce-scatter gradients
        reduce_scatter_gradients();
        
        optimizer.step();
        
        // Free full parameters
        full_params.clear();
        
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
    
    std::cout << "Starting FSDP Training - Rank " << rank << "/" << world_size << std::endl;
    
    // Set device
    torch::Device device(torch::kCUDA, local_rank);
    
    // Create model
    auto model = std::make_shared<TransformerModel>(
        vocab_size, d_model, nhead, num_layers);
    model->to(device);
    
    // Initialize FSDP trainer
    FSDPTrainer trainer(model, rank, world_size, "localhost:29500");
    
    // Create optimizer
    torch::optim::Adam optimizer(model->parameters(), 
                                 torch::optim::AdamOptions(1e-4));
    
    // Load checkpoint if exists
    int start_epoch = 0;
    load_checkpoint("checkpoint_fsdp.pt", model, optimizer, start_epoch);
    
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
        save_checkpoint("checkpoint_fsdp.pt", model, optimizer, epoch, rank);
    }
    
    if (rank == 0) {
        std::cout << "\nFSDP Training complete!" << std::endl;
    }
    
    return 0;
}
