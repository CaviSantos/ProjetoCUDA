#include "laplace_filter.h"
#include "laplace_filter_kernel.h"

#include <stdexcept>

Image cuda::LaplaceFilter(Image image) {
  Image output = image;
  output.data = new uint8_t[image.height * image.width * image.channels];

  kernel::LaplaceFilter(image, &output);

  return output;
}