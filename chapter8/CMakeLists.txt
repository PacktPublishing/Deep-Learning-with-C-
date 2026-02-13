cmake_minimum_required(VERSION 3.18)
project(autoencoder)

set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)

# Original autoencoder
add_executable(autoencoder autoencoder.cpp)
target_link_libraries(autoencoder "${TORCH_LIBRARIES}")

# Variational autoencoder
add_executable(vae vae.cpp)
target_link_libraries(vae "${TORCH_LIBRARIES}")