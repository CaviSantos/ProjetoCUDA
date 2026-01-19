#include "gaussian_blur.h"
#include "gaussian_blur_kernel.h"
#include <stdexcept>

Image cuda::GaussianBlur(Image image) {
  Image output = image;
  output.data = new uint8_t[image.height * image.width * image.channels];

  kernel::GaussianBlur(image, &output);

  return output;
}