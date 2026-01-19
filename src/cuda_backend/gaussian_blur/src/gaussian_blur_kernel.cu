#include <cuda_runtime.h>

#include "gaussian_blur_kernel.h"

__constant__ float g_kernel[3][3] = {{0.0625f, 0.125f, 0.0625f},
                                     {0.125f, 0.25f, 0.125f},
                                     {0.0625f, 0.125f, 0.0625f}};

__global__ void GaussianBlur(uint8_t *input, uint8_t *output, int height,
                             int width, int channels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {

    for (int c = 0; c < channels; c++) {
      float blurred_val = 0.0f;

      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {

          int cur_y = min(max(y + ky, 0), height - 1);
          int cur_x = min(max(x + kx, 0), width - 1);

          int neighbor_idx = (cur_y * width + cur_x) * channels + c;

          blurred_val += input[neighbor_idx] * g_kernel[ky + 1][kx + 1];
        }
      }

      int out_idx = (y * width + x) * channels + c;
      output[out_idx] = (uint8_t)blurred_val;
    }
  }
}

void cuda::kernel::GaussianBlur(Image input_image, Image *output_image) {
  size_t image_size =
      input_image.height * input_image.width * input_image.channels;

  uint8_t *d_input;
  uint8_t *d_output;

  cudaMalloc(&d_input, image_size);
  cudaMalloc(&d_output, image_size);

  cudaMemcpy(d_input, input_image.data, image_size, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks(
      (input_image.width + threadsPerBlock.x - 1) / threadsPerBlock.x,
      (input_image.height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  const int passes = 5;

  for (int i = 0; i < passes; i++) {
    ::GaussianBlur<<<numBlocks, threadsPerBlock>>>(
        d_input, d_output, input_image.height, input_image.width,
        input_image.channels);
    cudaDeviceSynchronize();

    // Swap pointers: The output of this pass becomes the input for the next
    uint8_t *temp = d_input;
    d_input = d_output;
    d_output = temp;
  }

  cudaMemcpy(output_image->data, d_input, image_size, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
}