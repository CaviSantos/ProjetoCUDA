#ifndef CUDA_BACKEND_RGB_TO_GRAY_INCLUDE_RGB_TO_GRAY_KERNEL_H
#define CUDA_BACKEND_RGB_TO_GRAY_INCLUDE_RGB_TO_GRAY_KERNEL_H

#include "image.h"

namespace cuda {
namespace kernel {

void RgbToGray(Image input_image, Image *output_image);

}
} // namespace cuda

#endif