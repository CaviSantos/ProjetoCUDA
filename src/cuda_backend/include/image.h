#ifndef CUDA_BACKEND_INCLUDE_IMAGE_H
#define CUDA_BACKEND_INCLUDE_IMAGE_H

#include <stdexcept>

struct Image {
  uint8_t *data;
  size_t height;
  size_t width;
  size_t channels;
};

#endif