#ifndef CUDA_BACKEND_LAPLACE_FILTER_INCLUDE_LAPLACE_FILTER_H
#define CUDA_BACKEND_LAPLACE_FILTER_INCLUDE_LAPLACE_FILTER_H

#include "image.h"

namespace cuda {
Image LaplaceFilter(Image image);
}

#endif