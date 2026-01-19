#include <cuda_runtime.h>

#include "rgb_to_gray_kernel.h"

__device__ constexpr float gray_red_weight = 0.2989;
__device__ constexpr float gray_green_weight = 0.5870;
__device__ constexpr float gray_blue_weight = 0.1140;

__global__ void RgbToGray(uint8_t *input, uint8_t *output, uint8_t channels) {
  int pixel_idx = blockDim.x * blockIdx.x + threadIdx.x;

  uint8_t *input_ptr = &input[pixel_idx * channels];

  uint8_t red = input_ptr[0];
  uint8_t green = input_ptr[1];
  uint8_t blue = input_ptr[2];

  output[pixel_idx] =
      (uint8_t)(red * gray_red_weight + green * gray_green_weight +
                blue * gray_blue_weight);
}

void cuda::kernel::RgbToGray(Image input_image, Image *output_image) {
  size_t input_image_size =
      input_image.height * input_image.width * input_image.channels;
  size_t output_image_size = input_image.height * input_image.width;

  uint8_t *d_input;
  uint8_t *d_output;

  cudaMalloc(&d_input, input_image_size);
  cudaMalloc(&d_output, output_image_size);

  cudaMemcpy(d_input, input_image.data, input_image_size,
             cudaMemcpyHostToDevice);
  ::RgbToGray<<<input_image.height, input_image.width>>>(d_input, d_output,
                                                         input_image.channels);
  cudaDeviceSynchronize();

  cudaMemcpy(output_image->data, d_output, output_image_size,
             cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}