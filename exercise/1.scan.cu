#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define DIM (1024)
#define BLOCK_DIM DIM
#define STRIDE_FACTOR 2

// CPU version of the scan kernel EXCLUSIVE
void scan_cpu(const float *input, float *output, const int length)
{
  output[0] = 0;
  for (int i = 1; i < length; ++i)
  {
    output[i] = output[i - 1] + input[i - 1];
  }
}

__global__ void scan_gpu(const float *__restrict__ input, float *__restrict__ output, const int length)
{
  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum = 0;

  for (unsigned int i = 0; i < tid; i++)
  {
    sum += input[i];
  }

  output[tid] = sum;

  return;
}

int main()
{
  std::vector<float> input(DIM);
  std::vector<float> output(DIM);
  for (int i = 0; i < DIM; ++i)
    input[i] = static_cast<float>(rand()) / RAND_MAX; // Random value between 0 and 1 }

  // CPU version
  scan_cpu(input.data(), output.data(), input.size());

  // GPU version
  float *input_d, *output_d;
  cudaMalloc(&input_d, DIM * sizeof(float));
  cudaMalloc(&output_d, DIM * sizeof(float));
  cudaMemcpy(input_d, input.data(), DIM * sizeof(float), cudaMemcpyHostToDevice);

  const dim3 block_dim(BLOCK_DIM, 1, 1);
  const dim3 grid_dim((DIM - 1) / BLOCK_DIM + 1, 1, 1);
  scan_gpu<<<grid_dim, block_dim>>>(input_d, output_d, DIM);

  cudaDeviceSynchronize();

  std::vector<float> gpu_results(DIM);
  cudaMemcpy(gpu_results.data(), output_d, DIM * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < gpu_results.size(); ++i)
  {
    if (std::abs(output[i] - gpu_results[i]) > 1e-3)
    {
      std::cout << "Scan CPU and GPU are NOT equivalent!" << std::endl;
      std::cout << "Index: " << i << std::endl;
      std::cout << "CPU: " << output[i] << std::endl;
      std::cout << "GPU: " << gpu_results[i] << std::endl;
      return EXIT_FAILURE;
    }
  }

  std::cout << "Scan CPU and GPU are equivalent!" << std::endl;

  // Cleanup
  cudaDeviceReset();

  return 0;
}
