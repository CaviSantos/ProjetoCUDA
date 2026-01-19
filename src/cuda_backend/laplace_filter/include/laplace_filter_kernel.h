#ifndef CUDA_BACKEND_LAPLACE_FILTER_INCLUDE_LAPLACE_FILTER_KERNEL_H
#define CUDA_BACKEND_LAPLACE_FILTER_INCLUDE_LAPLACE_FILTER_KERNEL_H

#include "image.h"

namespace cuda {
namespace kernel {

void LaplaceFilter(Image input_image, Image *output_image);

}
} // namespace cuda

#endif