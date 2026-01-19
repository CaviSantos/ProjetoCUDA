set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE STRING "Position Independent Code" FORCE)

project(cuda_backend LANGUAGES CXX CUDA)

add_library(${PROJECT_NAME} INTERFACE)

find_package(CUDAToolkit REQUIRED)

target_sources(${PROJECT_NAME}
  INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/include/image.h
)

target_include_directories(${PROJECT_NAME}
  INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
  INTERFACE
  CUDA::cudart
  gaussian_blur
  laplace_filter
  rgb_to_gray
)

add_subdirectory(gaussian_blur)
add_subdirectory(laplace_filter)
add_subdirectory(rgb_to_gray)
