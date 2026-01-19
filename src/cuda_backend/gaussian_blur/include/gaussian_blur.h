#ifndef CUDA_BACKEND_GAUSSIAN_BLUR_INCLUDE_GAUSSIAN_BLUR_H
#define CUDA_BACKEND_GAUSSIAN_BLUR_INCLUDE_GAUSSIAN_BLUR_H

#include "image.h"

namespace cuda {
Image GaussianBlur(Image image);
}

#endif