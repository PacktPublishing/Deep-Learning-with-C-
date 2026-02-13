#include <torch/torch.h>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/FileStore.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>

// ============================================================================
// Distributed Sampler: Partitions dataset across multiple processes
// ============================================================================
class DistributedSampler {
private:
    size_t num_samples;
    int rank;
    int world_size;
    size_t num_samples_per_rank;
    std::vector<size_t> indices;
    bool shuffle;
    unsigned seed;
    
public:
    DistributedSampler(size_t total_samples, int rank, int world_size, 
                       bool shuffle = true, unsigned seed = 0)
        : num_samples(total_samples), rank(rank), world_size(world_size),
          shuffle(shuffle), seed(seed) {
        num_samples_per_rank = (num_samples + world_size - 1) / world_size;
        std::cout << "[Rank " << rank << "] Sampler initialized: "
                  << num_samples_per_rank << " samples per rank" << std::endl;
    }
    
    void set_epoch(int epoch) {
        indices.clear();
        std::vector<size_t> all_indices(num_samples);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        if (shuffle) {
            std::mt19937 gen(seed + epoch);
            std::shuffle(all_indices.begin(), all_indices.end(), gen);
        }
        
        size_t total_size = num_samples_per_rank * world_size;
        while (all_indices.size() < total_size) {
            all_indices.push_back(all_indices[all_indices.size() % num_samples]);
        }
        
        for (size_t i = rank; i < all_indices.size(); i += world_size) {
            indices.push_back(all_indices[i]);
        }
    }
    
    const std::vector<size_t>& get_indices() const { return indices; }
    size_t size() const { return num_samples_per_rank; }
};

// ============================================================================
// Simple Neural Network Model Definition
// ============================================================================
struct SimpleNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    
    SimpleNet(int64_t input_size, int64_t hidden_size, int64_t num_classes) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, num_classes));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};

// ============================================================================
// Fully Sharded Data Parallel (FSDP) Trainer Class
// ============================================================================
// FSDP shards model parameters, gradients, and optimizer states across GPUs
// Key difference from DDP: Each GPU only stores a fraction of the model
class FSDPTrainer {
private:
    std::shared_ptr<torch::nn::Module> model;
    std::shared_ptr<c10d::ProcessGroup> process_group;
    int rank;
    int world_size;
    
    // FSDP-specific: Store sharded parameters
    // Each rank owns a subset of the full model parameters
    std::vector<torch::Tensor> sharded_params;
    std::vector<torch::Tensor> full_params;  // Temporarily gathered during forward/backward
    
public:
    FSDPTrainer(std::shared_ptr<torch::nn::Module> model_ptr, 
                int rank, int world_size, const std::string& master_addr)
        : model(model_ptr), rank(rank), world_size(world_size) {
        
        // ====================================================================
        // Step 1: Initialize Process Group
        // ====================================================================
        auto store = std::make_shared<c10d::FileStore>(master_addr, world_size);
        c10d::ProcessGroupNCCL::Options options;
        process_group = std::make_shared<c10d::ProcessGroupNCCL>(
            store, rank, world_size, options);
        
        // ====================================================================
        // Step 2: Broadcast initial parameters from rank 0
        // ====================================================================
        // CRITICAL: Ensure all ranks start with same model weights
        broadcast_parameters();
        
        // ====================================================================
        // Step 3: Shard Model Parameters Across Ranks
        // ====================================================================
        // Unlike DDP where each GPU has full model copy,
        // FSDP splits parameters across GPUs to save memory
        shard_parameters();
        
        std::cout << "[Rank " << rank << "] FSDP Trainer initialized" << std::endl;
    }
    
    // ========================================================================
    // Broadcast Parameters: Ensure all ranks start with same weights
    // ========================================================================
    void broadcast_parameters() {
        for (auto& param : model->parameters()) {
            std::vector<torch::Tensor> tensors = {param.data()};
            c10d::BroadcastOptions opts;
            opts.rootRank = 0;  // Rank 0 is source
            process_group->broadcast(tensors, opts)->wait();
        }
        
        if (rank == 0) {
            std::cout << "[Rank 0] Broadcasted initial parameters to all ranks" << std::endl;
        }
    }
    
    // ========================================================================
    // Shard Parameters: Split model parameters across all ranks
    // ========================================================================
    // Each rank stores only a portion of the model to reduce memory usage
    // Example: 1000 params, 4 GPUs -> each GPU stores 250 params
    void shard_parameters() {
        std::vector<torch::Tensor> all_params;
        for (const auto& param : model->parameters()) {
            all_params.push_back(param.data().flatten());
        }
        
        // Concatenate all parameters into single tensor
        torch::Tensor concat_params = torch::cat(all_params);
        int64_t total_size = concat_params.numel();
        
        // Calculate shard size for this rank
        int64_t shard_size = (total_size + world_size - 1) / world_size;
        int64_t start_idx = rank * shard_size;
        int64_t end_idx = std::min(start_idx + shard_size, total_size);
        
        // Extract this rank's shard
        if (start_idx < total_size) {
            torch::Tensor shard = concat_params.slice(0, start_idx, end_idx).clone();
            sharded_params.push_back(shard);
        }
        
        std::cout << "[Rank " << rank << "] Sharded " << (end_idx - start_idx) 
                  << " / " << total_size << " parameters" << std::endl;
    }
    
    // ========================================================================
    // All-Gather Parameters: Collect all shards to reconstruct full parameters
    // ========================================================================
    // Before forward/backward pass, gather all parameter shards from all ranks
    // This temporarily reconstructs the full model on each GPU
    void all_gather_parameters() {
        full_params.clear();
        
        for (const auto& shard : sharded_params) {
            // Prepare tensor list to receive shards from all ranks
            std::vector<torch::Tensor> gathered_shards;
            for (int i = 0; i < world_size; ++i) {
                gathered_shards.push_back(torch::empty_like(shard));
            }
            
            // All-gather operation: collect shards from all ranks
            std::vector<std::vector<torch::Tensor>> output_tensors = {gathered_shards};
            std::vector<torch::Tensor> input_tensors = {shard};
            
            c10d::AllgatherOptions opts;
            process_group->allgather(output_tensors, input_tensors, opts)->wait();
            
            // Concatenate all shards to form full parameter
            torch::Tensor full_param = torch::cat(gathered_shards);
            full_params.push_back(full_param);
        }
        
        // Update model with gathered parameters
        update_model_params(full_params);
    }
    
    // ========================================================================
    // Update Model Parameters: Copy gathered parameters back to model
    // ========================================================================
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
    
    // ========================================================================
    // Reduce-Scatter Gradients: Aggregate and distribute gradient shards
    // ========================================================================
    // After backward pass, reduce gradients and scatter back to respective ranks
    // Each rank receives only its shard of the reduced gradients
    void reduce_scatter_gradients() {
        std::vector<torch::Tensor> all_grads;
        for (const auto& param : model->parameters()) {
            if (param.grad().defined()) {
                all_grads.push_back(param.grad().flatten());
            }
        }
        
        if (all_grads.empty()) return;
        
        // Concatenate all gradients
        torch::Tensor concat_grads = torch::cat(all_grads);
        int64_t total_size = concat_grads.numel();
        int64_t shard_size = (total_size + world_size - 1) / world_size;
        
        // Split gradients into chunks for each rank
        std::vector<torch::Tensor> input_list;
        for (int i = 0; i < world_size; ++i) {
            int64_t start = i * shard_size;
            int64_t end = std::min(start + shard_size, total_size);
            if (start < total_size) {
                input_list.push_back(concat_grads.slice(0, start, end));
            }
        }
        
        // Reduce-scatter: sum gradients and distribute shards
        torch::Tensor output = torch::empty_like(input_list[rank]);
        std::vector<torch::Tensor> output_list = {output};
        
        c10d::ReduceScatterOptions opts;
        opts.reduceOp = c10d::ReduceOp::SUM;
        process_group->reduce_scatter(output_list, {input_list}, opts)->wait();
        
        // Average the reduced gradients
        output.div_(world_size);
    }
    
    // ========================================================================
    // Training Step: FSDP forward-backward-update cycle
    // ========================================================================
    void train_step(torch::Tensor input, torch::Tensor target,
                   torch::optim::Optimizer& optimizer) {
        // ====================================================================
        // Step 1: All-Gather - Reconstruct full model from shards
        // ====================================================================
        all_gather_parameters();
        
        // ====================================================================
        // Step 2: Forward Pass - Compute loss with full model
        // ====================================================================
        auto output = model->forward(input);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        
        std::cout << "[Rank " << rank << "] Loss: " << loss.item<float>() << std::endl;
        
        // ====================================================================
        // Step 3: Backward Pass - Compute gradients
        // ====================================================================
        optimizer.zero_grad();
        loss.backward();
        
        // ====================================================================
        // Step 4: Reduce-Scatter - Aggregate gradients and distribute shards
        // ====================================================================
        reduce_scatter_gradients();
        
        // ====================================================================
        // Step 5: Optimizer Update
        // ====================================================================
        optimizer.step();
        
        // ====================================================================
        // Step 6: Free full parameters to save memory
        // ====================================================================
        // After backward, we only need sharded params for optimizer update
        full_params.clear();
    }
    
    // ========================================================================
    // Training Loop
    // ========================================================================
    void train(torch::optim::Optimizer& optimizer, 
               const std::vector<torch::Tensor>& train_data,
               const std::vector<torch::Tensor>& train_labels,
               DistributedSampler& sampler,
               int num_epochs) {
        
        model->train();
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            sampler.set_epoch(epoch);
            const auto& indices = sampler.get_indices();
            
            std::cout << "\n[Rank " << rank << "] Epoch " << epoch + 1 
                      << "/" << num_epochs << " - Processing " 
                      << indices.size() << " samples" << std::endl;
            
            // FSDP: Different batches per GPU (data parallelism)
            // Model is sharded for memory, but each GPU processes different data
            for (size_t idx : indices) {
                size_t data_idx = idx % train_data.size();
                train_step(train_data[data_idx], train_labels[data_idx], optimizer);
            }
        }
    }
};

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char* argv[]) {
    int rank = 0;
    int world_size = 2;
    std::string store_path = "/tmp/fsdp_store";
    
    if (argc >= 3) {
        rank = std::stoi(argv[1]);
        world_size = std::stoi(argv[2]);
    }
    
    std::cout << "Starting FSDP Training - Rank: " << rank 
              << ", World Size: " << world_size << std::endl;
    
    // Model setup
    const int64_t input_size = 784;
    const int64_t hidden_size = 128;
    const int64_t num_classes = 10;
    
    auto model = std::make_shared<SimpleNet>(input_size, hidden_size, num_classes);
    
    torch::Device device(torch::kCUDA, rank);
    model->to(device);
    
    // FSDP initialization
    FSDPTrainer trainer(model, rank, world_size, store_path);
    
    // Optimizer
    torch::optim::SGD optimizer(
        model->parameters(), 
        torch::optim::SGDOptions(0.01)
    );
    
    // Data preparation
    const size_t total_samples = 100;
    std::vector<torch::Tensor> train_data;
    std::vector<torch::Tensor> train_labels;
    
    for (size_t i = 0; i < total_samples; ++i) {
        train_data.push_back(torch::randn({input_size}).to(device));
        train_labels.push_back(torch::tensor(i % num_classes).to(device));
    }
    
    // Distributed sampler for data parallelism
    DistributedSampler sampler(total_samples, rank, world_size, true, 42);
    
    std::cout << "[Rank " << rank << "] Dataset size: " << total_samples 
              << ", Samples per rank: " << sampler.size() << std::endl;
    
    // Training
    int num_epochs = 3;
    trainer.train(optimizer, train_data, train_labels, sampler, num_epochs);
    
    std::cout << "\n[Rank " << rank << "] Training Complete!" << std::endl;
    
    return 0;
}
