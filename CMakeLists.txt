cmake_minimum_required(VERSION 3.22)

project(
  cuda_image_processor
  VERSION 1.0.0
  LANGUAGES CXX
)

add_subdirectory(src/cuda_backend)

# Find pybind11 and Python
find_package(pybind11 3.0.1 CONFIG REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
pybind11_add_module(_core MODULE src/main.cpp)
target_link_libraries(_core
  PRIVATE 
  pybind11::headers
  cuda_backend
)

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION ${PROJECT_NAME})