#ifndef CUDA_BACKEND_GAUSSIAN_BLUR_INCLUDE_GAUSSIAN_BLUR_KERNEL_H
#define CUDA_BACKEND_GAUSSIAN_BLUR_INCLUDE_GAUSSIAN_BLUR_KERNEL_H

#include "image.h"

namespace cuda {
namespace kernel {

void GaussianBlur(Image input_image, Image *output_image);

}
} // namespace cuda

#endif