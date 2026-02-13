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
// Ensures each GPU processes different data samples without overlap
class DistributedSampler {
private:
    size_t num_samples;      // Total number of samples in dataset
    int rank;                // Current process rank
    int world_size;          // Total number of processes
    size_t num_samples_per_rank;  // Samples assigned to each rank
    std::vector<size_t> indices;  // Shuffled indices for this rank
    bool shuffle;            // Whether to shuffle data each epoch
    unsigned seed;           // Random seed for reproducibility
    
public:
    DistributedSampler(size_t total_samples, int rank, int world_size, 
                       bool shuffle = true, unsigned seed = 0)
        : num_samples(total_samples), rank(rank), world_size(world_size),
          shuffle(shuffle), seed(seed) {
        
        // Calculate samples per rank (pad if necessary to ensure equal distribution)
        num_samples_per_rank = (num_samples + world_size - 1) / world_size;
        
        std::cout << "[Rank " << rank << "] Sampler initialized: "
                  << num_samples_per_rank << " samples per rank" << std::endl;
    }
    
    // Generate indices for current epoch
    // Each rank gets a non-overlapping subset of the data
    void set_epoch(int epoch) {
        indices.clear();
        
        // Create full index list
        std::vector<size_t> all_indices(num_samples);
        std::iota(all_indices.begin(), all_indices.end(), 0);
        
        // Shuffle if enabled (using epoch as seed for reproducibility)
        if (shuffle) {
            std::mt19937 gen(seed + epoch);
            std::shuffle(all_indices.begin(), all_indices.end(), gen);
        }
        
        // Pad indices to make it evenly divisible by world_size
        size_t total_size = num_samples_per_rank * world_size;
        while (all_indices.size() < total_size) {
            all_indices.push_back(all_indices[all_indices.size() % num_samples]);
        }
        
        // Extract indices for this rank
        // Rank 0 gets indices [0, world_size, 2*world_size, ...]
        // Rank 1 gets indices [1, world_size+1, 2*world_size+1, ...]
        for (size_t i = rank; i < all_indices.size(); i += world_size) {
            indices.push_back(all_indices[i]);
        }
    }
    
    // Get indices for this rank
    const std::vector<size_t>& get_indices() const {
        return indices;
    }
    
    size_t size() const {
        return num_samples_per_rank;
    }
};

// ============================================================================
// Simple Neural Network Model Definition
// ============================================================================
// Define a basic feedforward neural network for demonstration
struct SimpleNet : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    
    SimpleNet(int64_t input_size, int64_t hidden_size, int64_t num_classes) {
        // Layer 1: input -> hidden
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        // Layer 2: hidden -> hidden
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        // Layer 3: hidden -> output
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, num_classes));
    }
    
    // Forward pass through the network
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};

// ============================================================================
// Distributed Data Parallel Trainer Class
// ============================================================================
// Handles multi-GPU training with gradient synchronization via NCCL
class DDPTrainer {
private:
    std::shared_ptr<torch::nn::Module> model;           // Neural network model
    std::shared_ptr<c10d::ProcessGroup> process_group;  // Communication group for distributed ops
    int rank;                                            // Current process rank (GPU ID)
    int world_size;                                      // Total number of processes (GPUs)
    
public:
    // Constructor: Initialize DDP trainer with model and distributed settings
    // - model_ptr: Shared pointer to the neural network model
    // - rank: Current process ID (0 to world_size-1)
    // - world_size: Total number of distributed processes
    // - master_addr: Address for process coordination (file path for FileStore)
    DDPTrainer(std::shared_ptr<torch::nn::Module> model_ptr, 
               int rank, int world_size, const std::string& master_addr)
        : model(model_ptr), rank(rank), world_size(world_size) {
        
        // ====================================================================
        // Step 1: Initialize Process Group for Inter-Process Communication
        // ====================================================================
        // FileStore: Shared file-based key-value store for process coordination
        // All processes read/write to this file to exchange initialization info
        auto store = std::make_shared<c10d::FileStore>(master_addr, world_size);
        
        // NCCL (NVIDIA Collective Communications Library) options
        // NCCL provides optimized GPU-to-GPU communication primitives
        c10d::ProcessGroupNCCL::Options options;
        
        // Create NCCL process group for collective operations (broadcast, all-reduce, etc.)
        process_group = std::make_shared<c10d::ProcessGroupNCCL>(
            store, rank, world_size, options);
        
        // ====================================================================
        // Step 2: Synchronize Initial Model Parameters Across All Processes
        // ====================================================================
        // Ensures all GPUs start with identical model weights
        broadcast_parameters();
        
        std::cout << "[Rank " << rank << "] DDP Trainer initialized" << std::endl;
    }
     
    // ========================================================================
    // Broadcast Parameters: Synchronize model weights from rank 0 to all ranks
    // ========================================================================
    // Called during initialization to ensure all processes have the same starting model
    void broadcast_parameters() {
        // Collect all model parameters (weights and biases)
        std::vector<torch::Tensor> params;
        for (const auto& param : model->parameters()) {
            params.push_back(param.data());
        }
        
        // Broadcast each parameter tensor from rank 0 to all other ranks
        for (auto& param : params) {
            std::vector<torch::Tensor> tensor_list = {param};
            
            // Configure broadcast options
            c10d::BroadcastOptions opts;
            opts.rootRank = 0;  // Rank 0 is the source of truth
            
            // Perform broadcast and wait for completion
            // After this, all ranks have the same parameter values
            process_group->broadcast(tensor_list, opts)->wait();
        }
        
        std::cout << "[Rank " << rank << "] Parameters broadcasted" << std::endl;
    }
    
    // ========================================================================
    // All-Reduce Gradients: Average gradients across all processes
    // ========================================================================
    // Core DDP operation - synchronizes gradients after backward pass
    // Each GPU computes gradients on its local batch, then averages them
    void all_reduce_gradients() {
        // Collect all gradient tensors from model parameters
        std::vector<torch::Tensor> gradients;
        for (const auto& param : model->parameters()) {
            // Only process parameters that have gradients (some may be frozen)
            if (param.grad().defined()) {
                gradients.push_back(param.grad());
            }
        }
        
        // All-reduce operation: Sum gradients from all ranks, then average
        for (auto& grad : gradients) {
            std::vector<torch::Tensor> tensor_list = {grad};
            
            // Configure all-reduce options
            c10d::AllreduceOptions opts;
            opts.reduceOp = c10d::ReduceOp::SUM;  // Sum gradients across all ranks
            
            // Perform all-reduce: each rank gets the sum of all gradients
            process_group->allreduce(tensor_list, opts)->wait();
            
            // Average the summed gradients by dividing by world_size
            // This gives us the mean gradient across all data batches
            grad.div_(world_size);
        }
    }
    
    // ========================================================================
    // Training Step: Complete forward-backward-update cycle
    // ========================================================================
    // Performs one iteration of distributed training
    void train_step(torch::Tensor input, torch::Tensor target,
                   torch::optim::Optimizer& optimizer) {
        // ====================================================================
        // Step 1: Forward Pass - Compute predictions and loss
        // ====================================================================
        // Each GPU processes its own batch independently
        auto output = model->forward(input);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        
        std::cout << "[Rank " << rank << "] Loss: " << loss.item<float>() << std::endl;
        
        // ====================================================================
        // Step 2: Backward Pass - Compute gradients
        // ====================================================================
        // Clear previous gradients
        optimizer.zero_grad();
        
        // Backpropagate to compute gradients
        loss.backward();
        
        // ====================================================================
        // Step 3: Gradient Synchronization - Average gradients across GPUs
        // ====================================================================
        // This is the key DDP operation that keeps models synchronized
        all_reduce_gradients();
        
        // ====================================================================
        // Step 4: Parameter Update - Apply averaged gradients
        // ====================================================================
        // All GPUs now have identical gradients, so they update identically
        // This keeps all model replicas in sync
        optimizer.step();
    }
    
    // ========================================================================
    // Training Loop: Run multiple epochs of distributed training
    // ========================================================================
    void train(torch::optim::Optimizer& optimizer, 
               const std::vector<torch::Tensor>& train_data,
               const std::vector<torch::Tensor>& train_labels,
               DistributedSampler& sampler,
               int num_epochs) {
        
        model->train();  // Set model to training mode
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Set epoch for sampler to shuffle data differently each epoch
            sampler.set_epoch(epoch);
            const auto& indices = sampler.get_indices();
            
            std::cout << "\n[Rank " << rank << "] Epoch " << epoch + 1 
                      << "/" << num_epochs << " - Processing " 
                      << indices.size() << " samples" << std::endl;
            
            // Each rank processes its assigned subset of data
            for (size_t idx : indices) {
                // Ensure index is within bounds (handles padding)
                size_t data_idx = idx % train_data.size();
                train_step(train_data[data_idx], train_labels[data_idx], optimizer);
            }
        }
    }
};

// ============================================================================
// Main Function: Entry point for DDP training
// ============================================================================
int main(int argc, char* argv[]) {
    // ========================================================================
    // Configuration: Set up distributed training parameters
    // ========================================================================
    // In real deployment, these would come from environment variables or CLI args
    int rank = 0;              // Current process rank (GPU ID)
    int world_size = 2;        // Total number of GPUs
    std::string store_path = "/tmp/ddp_store";  // Shared coordination file
    
    // Parse command line arguments if provided
    if (argc >= 3) {
        rank = std::stoi(argv[1]);
        world_size = std::stoi(argv[2]);
    }
    
    std::cout << "Starting DDP Training - Rank: " << rank 
              << ", World Size: " << world_size << std::endl;
    
    // ========================================================================
    // Model Setup: Create and move model to GPU
    // ========================================================================
    const int64_t input_size = 784;      // Example: MNIST image size (28x28)
    const int64_t hidden_size = 128;     // Hidden layer size
    const int64_t num_classes = 10;      // Number of output classes
    
    auto model = std::make_shared<SimpleNet>(input_size, hidden_size, num_classes);
    
    // Move model to GPU corresponding to this rank
    torch::Device device(torch::kCUDA, rank);
    model->to(device);
    
    // ========================================================================
    // DDP Initialization: Create distributed trainer
    // ========================================================================
    DDPTrainer trainer(model, rank, world_size, store_path);
    
    // ========================================================================
    // Optimizer Setup: Configure SGD optimizer
    // ========================================================================
    torch::optim::SGD optimizer(
        model->parameters(), 
        torch::optim::SGDOptions(0.01)  // Learning rate
    );
    
    // ========================================================================
    // Data Preparation: Create dummy training data
    // ========================================================================
    const size_t total_samples = 100;  // Total dataset size
    std::vector<torch::Tensor> train_data;
    std::vector<torch::Tensor> train_labels;
    
    // Generate synthetic data for demonstration
    for (size_t i = 0; i < total_samples; ++i) {
        train_data.push_back(torch::randn({input_size}).to(device));
        train_labels.push_back(torch::tensor(i % num_classes).to(device));
    }
    
    // ========================================================================
    // Distributed Sampler: Partition data across GPUs
    // ========================================================================
    // Each rank will process a non-overlapping subset of the data
    DistributedSampler sampler(total_samples, rank, world_size, true, 42);
    
    std::cout << "[Rank " << rank << "] Dataset size: " << total_samples 
              << ", Samples per rank: " << sampler.size() << std::endl;
    
    // ========================================================================
    // Training: Run distributed training
    // ========================================================================
    int num_epochs = 3;
    trainer.train(optimizer, train_data, train_labels, sampler, num_epochs);
    
    std::cout << "\n[Rank " << rank << "] Training Complete!" << std::endl;
    
    return 0;
}
