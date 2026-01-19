#include "rgb_to_gray.h"
#include "rgb_to_gray_kernel.h"
#include <stdexcept>

Image cuda::RgbToGray(Image image) {
  if (image.channels < 3) {
    throw std::runtime_error("Not an RGB image");
  }

  Image output = image;
  output.data = new uint8_t[image.height * image.width];
  output.channels = 1;

  kernel::RgbToGray(image, &output);

  return output;
}