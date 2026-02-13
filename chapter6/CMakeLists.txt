cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(LibTorchConv)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(libtorch_conv LibTorchConvLayer.cpp)
target_link_libraries(libtorch_conv "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_conv PROPERTY CXX_STANDARD 17)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET libtorch_conv
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:libtorch_conv>)
endif (MSVC)