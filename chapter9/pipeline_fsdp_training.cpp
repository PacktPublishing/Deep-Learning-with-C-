#include <torch/torch.h>
#include <c10d/ProcessGroupNCCL.hpp>
#include <c10d/FileStore.hpp>
#include <iostream>
#include <memory>
#include <vector>
#include <queue>

// ============================================================================
// Pipeline Stage: Single layer of the model
// ============================================================================
struct PipelineStage : torch::nn::Module {
    torch::nn::Linear layer{nullptr};
    int stage_id;
    
    PipelineStage(int64_t input_size, int64_t output_size, int id)
        : stage_id(id) {
        layer = register_module("layer", torch::nn::Linear(input_size, output_size));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        return torch::relu(layer->forward(x));
    }
};

// ============================================================================
// Pipeline + FSDP Trainer
// ============================================================================
class PipelineFSDPTrainer {
private:
    std::shared_ptr<PipelineStage> stage;
    std::shared_ptr<c10d::ProcessGroup> process_group;
    int rank;
    int world_size;
    int num_stages;
    
    // FSDP: Sharded parameters for this stage
    std::vector<torch::Tensor> sharded_params;
    std::vector<torch::Tensor> full_params;
    
    // Pipeline: Micro-batch queue
    std::queue<torch::Tensor> forward_queue;
    std::queue<torch::Tensor> backward_queue;
    
public:
    PipelineFSDPTrainer(std::shared_ptr<PipelineStage> stage_ptr,
                        int rank, int world_size, int num_stages,
                        const std::string& store_path)
        : stage(stage_ptr), rank(rank), world_size(world_size), 
          num_stages(num_stages) {
        
        auto store = std::make_shared<c10d::FileStore>(store_path, world_size);
        c10d::ProcessGroupNCCL::Options options;
        process_group = std::make_shared<c10d::ProcessGroupNCCL>(
            store, rank, world_size, options);
        
        shard_parameters();
        
        std::cout << "[Rank " << rank << "] Pipeline+FSDP initialized for stage " 
                  << stage->stage_id << std::endl;
    }
    
    // ========================================================================
    // FSDP: Shard parameters across ranks within same stage
    // ========================================================================
    void shard_parameters() {
        std::vector<torch::Tensor> all_params;
        for (const auto& param : stage->parameters()) {
            all_params.push_back(param.data().flatten());
        }
        
        if (all_params.empty()) return;
        
        torch::Tensor concat_params = torch::cat(all_params);
        int64_t total_size = concat_params.numel();
        
        // Shard across ranks in same pipeline stage
        int ranks_per_stage = world_size / num_stages;
        int stage_rank = rank % ranks_per_stage;
        
        int64_t shard_size = (total_size + ranks_per_stage - 1) / ranks_per_stage;
        int64_t start_idx = stage_rank * shard_size;
        int64_t end_idx = std::min(start_idx + shard_size, total_size);
        
        if (start_idx < total_size) {
            torch::Tensor shard = concat_params.slice(0, start_idx, end_idx).clone();
            sharded_params.push_back(shard);
        }
        
        std::cout << "[Rank " << rank << "] Stage " << stage->stage_id 
                  << " sharded " << (end_idx - start_idx) << "/" << total_size 
                  << " params" << std::endl;
    }
    
    // ========================================================================
    // FSDP: All-gather parameters within stage
    // ========================================================================
    void all_gather_parameters() {
        if (sharded_params.empty()) return;
        
        full_params.clear();
        int ranks_per_stage = world_size / num_stages;
        
        for (const auto& shard : sharded_params) {
            std::vector<torch::Tensor> gathered_shards;
            for (int i = 0; i < ranks_per_stage; ++i) {
                gathered_shards.push_back(torch::empty_like(shard));
            }
            
            // All-gather within stage group
            std::vector<std::vector<torch::Tensor>> output_tensors = {gathered_shards};
            std::vector<torch::Tensor> input_tensors = {shard};
            
            c10d::AllgatherOptions opts;
            process_group->allgather(output_tensors, input_tensors, opts)->wait();
            
            torch::Tensor full_param = torch::cat(gathered_shards);
            full_params.push_back(full_param);
        }
        
        update_stage_params(full_params);
    }
    
    // ========================================================================
    // Update stage parameters
    // ========================================================================
    void update_stage_params(const std::vector<torch::Tensor>& params) {
        if (params.empty()) return;
        
        torch::Tensor concat = torch::cat(params);
        int64_t offset = 0;
        
        for (auto& param : stage->parameters()) {
            int64_t numel = param.numel();
            torch::Tensor param_data = concat.slice(0, offset, offset + numel)
                                             .view(param.sizes());
            param.data().copy_(param_data);
            offset += numel;
        }
    }
    
    // ========================================================================
    // Pipeline: Send activation to next stage
    // ========================================================================
    void send_forward(torch::Tensor activation) {
        if (rank + 1 < world_size) {
            std::vector<torch::Tensor> send_tensors = {activation};
            c10d::SendOptions opts;
            process_group->send(send_tensors, rank + 1, 0)->wait();
        }
    }
    
    // ========================================================================
    // Pipeline: Receive activation from previous stage
    // ========================================================================
    torch::Tensor recv_forward() {
        if (rank > 0) {
            torch::Tensor activation = torch::empty({1, stage->layer->options.in_features()});
            std::vector<torch::Tensor> recv_tensors = {activation};
            c10d::RecvOptions opts;
            process_group->recv(recv_tensors, rank - 1, 0)->wait();
            return activation;
        }
        return torch::Tensor();
    }
    
    // ========================================================================
    // Pipeline: Send gradient to previous stage
    // ========================================================================
    void send_backward(torch::Tensor grad) {
        if (rank > 0) {
            std::vector<torch::Tensor> send_tensors = {grad};
            c10d::SendOptions opts;
            process_group->send(send_tensors, rank - 1, 1)->wait();
        }
    }
    
    // ========================================================================
    // Pipeline: Receive gradient from next stage
    // ========================================================================
    torch::Tensor recv_backward() {
        if (rank + 1 < world_size) {
            torch::Tensor grad = torch::empty({1, stage->layer->options.out_features()});
            std::vector<torch::Tensor> recv_tensors = {grad};
            c10d::RecvOptions opts;
            process_group->recv(recv_tensors, rank + 1, 1)->wait();
            return grad;
        }
        return torch::Tensor();
    }
    
    // ========================================================================
    // Pipeline Forward Pass
    // ========================================================================
    torch::Tensor pipeline_forward(torch::Tensor input) {
        // FSDP: Gather parameters for this stage
        all_gather_parameters();
        
        // Receive from previous stage (if not first stage)
        torch::Tensor x = (rank == 0) ? input : recv_forward();
        
        // Forward through this stage
        torch::Tensor output = stage->forward(x);
        
        // Send to next stage (if not last stage)
        if (rank + 1 < world_size) {
            send_forward(output.detach());
        }
        
        return output;
    }
    
    // ========================================================================
    // Pipeline Backward Pass
    // ========================================================================
    void pipeline_backward(torch::Tensor output, torch::Tensor target) {
        torch::Tensor loss;
        
        // Last stage computes loss
        if (rank == world_size - 1) {
            loss = torch::nn::functional::cross_entropy(output, target);
            std::cout << "[Rank " << rank << "] Loss: " << loss.item<float>() << std::endl;
            loss.backward();
        } else {
            // Receive gradient from next stage
            torch::Tensor grad_output = recv_backward();
            output.backward(grad_output);
        }
        
        // Send gradient to previous stage (if not first stage)
        if (rank > 0 && output.grad().defined()) {
            send_backward(output.grad());
        }
    }
    
    // ========================================================================
    // FSDP: Reduce-scatter gradients within stage
    // ========================================================================
    void reduce_scatter_gradients() {
        std::vector<torch::Tensor> all_grads;
        for (const auto& param : stage->parameters()) {
            if (param.grad().defined()) {
                all_grads.push_back(param.grad().flatten());
            }
        }
        
        if (all_grads.empty()) return;
        
        torch::Tensor concat_grads = torch::cat(all_grads);
        int64_t total_size = concat_grads.numel();
        
        int ranks_per_stage = world_size / num_stages;
        int stage_rank = rank % ranks_per_stage;
        int64_t shard_size = (total_size + ranks_per_stage - 1) / ranks_per_stage;
        
        std::vector<torch::Tensor> input_list;
        for (int i = 0; i < ranks_per_stage; ++i) {
            int64_t start = i * shard_size;
            int64_t end = std::min(start + shard_size, total_size);
            if (start < total_size) {
                input_list.push_back(concat_grads.slice(0, start, end));
            }
        }
        
        torch::Tensor output = torch::empty_like(input_list[stage_rank]);
        std::vector<torch::Tensor> output_list = {output};
        
        c10d::ReduceScatterOptions opts;
        opts.reduceOp = c10d::ReduceOp::SUM;
        process_group->reduce_scatter(output_list, {input_list}, opts)->wait();
        
        output.div_(ranks_per_stage);
    }
    
    // ========================================================================
    // Training Step
    // ========================================================================
    void train_step(torch::Tensor input, torch::Tensor target,
                   torch::optim::Optimizer& optimizer) {
        optimizer.zero_grad();
        
        // Pipeline forward
        torch::Tensor output = pipeline_forward(input);
        
        // Pipeline backward
        pipeline_backward(output, target);
        
        // FSDP: Reduce-scatter gradients
        reduce_scatter_gradients();
        
        // Update parameters
        optimizer.step();
        
        // Free full parameters
        full_params.clear();
    }
    
    // ========================================================================
    // Training Loop with Micro-batching
    // ========================================================================
    void train(torch::optim::Optimizer& optimizer,
               const std::vector<torch::Tensor>& train_data,
               const std::vector<torch::Tensor>& train_labels,
               int num_epochs, int num_microbatches) {
        
        stage->train();
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            std::cout << "\n[Rank " << rank << "] Epoch " << epoch + 1 
                      << "/" << num_epochs << std::endl;
            
            // Process data in micro-batches for pipeline efficiency
            for (size_t i = 0; i < train_data.size(); i += num_microbatches) {
                for (int mb = 0; mb < num_microbatches && i + mb < train_data.size(); ++mb) {
                    train_step(train_data[i + mb], train_labels[i + mb], optimizer);
                }
            }
        }
    }
};

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <rank> <world_size>" << std::endl;
        return 1;
    }
    
    int rank = std::stoi(argv[1]);
    int world_size = std::stoi(argv[2]);
    std::string store_path = "/tmp/pipeline_fsdp_store";
    
    std::cout << "Starting Pipeline+FSDP Training - Rank: " << rank 
              << ", World Size: " << world_size << std::endl;
    
    // Model configuration
    const int64_t input_size = 784;
    const int64_t hidden_size = 128;
    const int64_t num_classes = 10;
    const int num_stages = 3;  // 3 pipeline stages
    
    // Determine which stage this rank belongs to
    int stage_id = rank / (world_size / num_stages);
    
    // Create pipeline stage
    std::shared_ptr<PipelineStage> stage;
    if (stage_id == 0) {
        stage = std::make_shared<PipelineStage>(input_size, hidden_size, 0);
    } else if (stage_id == 1) {
        stage = std::make_shared<PipelineStage>(hidden_size, hidden_size, 1);
    } else {
        stage = std::make_shared<PipelineStage>(hidden_size, num_classes, 2);
    }
    
    torch::Device device(torch::kCUDA, rank);
    stage->to(device);
    
    // Initialize trainer
    PipelineFSDPTrainer trainer(stage, rank, world_size, num_stages, store_path);
    
    // Optimizer
    torch::optim::SGD optimizer(
        stage->parameters(),
        torch::optim::SGDOptions(0.01)
    );
    
    // Data preparation (simplified)
    const size_t total_samples = 100;
    std::vector<torch::Tensor> train_data;
    std::vector<torch::Tensor> train_labels;
    
    for (size_t i = 0; i < total_samples; ++i) {
        train_data.push_back(torch::randn({input_size}).to(device));
        train_labels.push_back(torch::tensor(i % num_classes).to(device));
    }
    
    std::cout << "[Rank " << rank << "] Stage " << stage_id 
              << " ready for training" << std::endl;
    
    // Training
    int num_epochs = 3;
    int num_microbatches = 4;
    trainer.train(optimizer, train_data, train_labels, num_epochs, num_microbatches);
    
    std::cout << "\n[Rank " << rank << "] Training Complete!" << std::endl;
    
    return 0;
}
