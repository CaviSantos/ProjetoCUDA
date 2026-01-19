cmake_minimum_required(VERSION 3.22.0)
project(laplace_filter LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME})

target_sources(${PROJECT_NAME}
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src/laplace_filter_kernel.cu
  ${CMAKE_CURRENT_LIST_DIR}/src/laplace_filter.cpp

  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include/laplace_filter_kernel.h
  ${CMAKE_CURRENT_LIST_DIR}/include/laplace_filter.h
)

target_link_libraries(${PROJECT_NAME} 
  PRIVATE
  cuda_backend
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
  src/cuda_backend
  ${CMAKE_CURRENT_LIST_DIR}/include
)