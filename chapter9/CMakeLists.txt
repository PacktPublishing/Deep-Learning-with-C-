cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(chapter9_transformer)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 17)

# Core architecture
add_executable(transformer transformer.cpp encoder.cpp decoder.cpp)
target_link_libraries(transformer "${TORCH_LIBRARIES}")

# Attention mechanisms
add_executable(attention attention.cpp)
add_executable(multihead_attention multihead_attention.cpp)
target_link_libraries(multihead_attention "${TORCH_LIBRARIES}")

add_executable(rope rope.cpp)
add_executable(simple_positional_encoding simple_positional_encoding.cpp)

# Distributed training
add_executable(ddp_training ddp_training.cpp)
target_link_libraries(ddp_training "${TORCH_LIBRARIES}")

add_executable(fsdp_training fsdp_training.cpp)
target_link_libraries(fsdp_training "${TORCH_LIBRARIES}")

add_executable(ddp_transformer ddp_transformer.cpp)
target_link_libraries(ddp_transformer "${TORCH_LIBRARIES}")

add_executable(fsdp_transformer fsdp_transformer.cpp)
target_link_libraries(fsdp_transformer "${TORCH_LIBRARIES}")

add_executable(pipeline_fsdp_training pipeline_fsdp_training.cpp)
target_link_libraries(pipeline_fsdp_training "${TORCH_LIBRARIES}")

# Model compression
add_executable(knowledge_distillation knowledge_distillation.cpp)
target_link_libraries(knowledge_distillation "${TORCH_LIBRARIES}")

add_executable(pruning pruning.cpp)
target_link_libraries(pruning "${TORCH_LIBRARIES}")

add_executable(gradient_pruning gradient_pruning.cpp)
target_link_libraries(gradient_pruning "${TORCH_LIBRARIES}")

add_executable(wanda_pruning wanda_pruning.cpp)
target_link_libraries(wanda_pruning "${TORCH_LIBRARIES}")

add_executable(transformer_pruning transformer_pruning.cpp)
target_link_libraries(transformer_pruning "${TORCH_LIBRARIES}")

add_executable(transformer_pruning_pipeline transformer_pruning_pipeline.cpp)
target_link_libraries(transformer_pruning_pipeline "${TORCH_LIBRARIES}")

add_executable(quantization quantization.cpp)
target_link_libraries(quantization "${TORCH_LIBRARIES}")

add_executable(low_rank_approximation low_rank_approximation.cpp)
target_link_libraries(low_rank_approximation "${TORCH_LIBRARIES}")

# Utilities
add_executable(training_utils training_utils.cpp)
target_link_libraries(training_utils "${TORCH_LIBRARIES}")

add_executable(data_loader data_loader.cpp)
target_link_libraries(data_loader "${TORCH_LIBRARIES}")

# Training examples
add_executable(bert_train bert_train.cpp encoder.cpp)
target_link_libraries(bert_train "${TORCH_LIBRARIES}")

add_executable(train_encoder train_encoder.cpp encoder.cpp)
target_link_libraries(train_encoder "${TORCH_LIBRARIES}")

add_executable(train_decoder train_decoder.cpp decoder.cpp)
target_link_libraries(train_decoder "${TORCH_LIBRARIES}")

add_executable(train_encoder_decoder train_encoder_decoder.cpp encoder.cpp decoder.cpp)
target_link_libraries(train_encoder_decoder "${TORCH_LIBRARIES}")

add_executable(torch_attention torch_attention.cpp)
target_link_libraries(torch_attention "${TORCH_LIBRARIES}")

add_executable(torch_encoder torch_encoder.cpp encoder.cpp)
target_link_libraries(torch_encoder "${TORCH_LIBRARIES}")

add_executable(torch_decoder torch_decoder.cpp decoder.cpp)
target_link_libraries(torch_decoder "${TORCH_LIBRARIES}")