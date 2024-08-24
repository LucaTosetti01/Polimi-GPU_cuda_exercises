#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

// #define DIM (512 + 256)
#define DIM 16
#define SIZE (DIM * DIM * DIM)
#define BLOCK_DIM 8

#define out(i, j, k) out[(i) * N * N + (j) * N + (k)]
#define in(i, j, k) in[(i) * N * N + (j) * N + (k)]

#define c0 1
#define c1 1
#define c2 1
#define c3 1
#define c4 1
#define c5 1
#define c6 1

#define CHECK(call)                                                                 \
  {                                                                                 \
    const cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess) {                                                       \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

// Function for CPU stencil computation
void stencil_cpu(const float *in, float *out, const int N)
{
    for (int i = 1; i < N - 1; ++i)
        for (int j = 1; j < N - 1; ++j)
            for (int k = 1; k < N - 1; ++k)
                out(i, j, k) = c0 * in(i, j, k) + c1 * in(i, j, k - 1) + c2 * in(i, j, k + 1) + c3 * in(i, j - 1, k) +
                               c4 * in(i, j + 1, k) + c5 * in(i - 1, j, k) + c6 * in(i + 1, j, k);
}

__global__ void stencil_gpu(const float *in, float *out, const int N)
{
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    int ty = blockDim.y * blockIdx.y + threadIdx.y;
    int tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= 1 && tx < N - 1 &&
        ty >= 1 && ty < N - 1 &&
        tz >= 1 && tz < N - 1)
        out(tx, ty, tz) = c0 * in(tx, ty, tz) + c1 * in(tx, ty, tz - 1) + c2 * in(tx, ty, tz + 1) + c3 * in(tx, ty - 1, tz) +
                          c4 * in(tx, ty + 1, tz) + c5 * in(tx - 1, ty, tz) + c6 * in(tx + 1, ty, tz);
}

int main()
{
   static_assert(DIM/BLOCK_DIM, "The dimension should be divisible by the number of output block computer by each block");
  // Generate random input data
  std::vector<float> input_data(SIZE);
  for (int i = 0; i < SIZE; ++i) { input_data[i] = rand() % 10; }

    // Compute stencil on CPU
    std::vector<float> cpu_result(SIZE);
    stencil_cpu(input_data.data(), cpu_result.data(), DIM);

    float *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float) * SIZE);
    cudaMalloc(&d_output, sizeof(float) * SIZE);
    cudaMemcpy(d_input, input_data.data(), sizeof(float)*SIZE, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 blocksPerGrid((DIM + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (DIM + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (DIM + threadsPerBlock.z - 1) / threadsPerBlock.z);
    stencil_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, DIM);
    cudaDeviceSynchronize();
    std::vector<float> gpu_results(SIZE);
    cudaMemcpy(gpu_results.data(), d_output, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++)
    {
        if (cpu_result[i] != gpu_results[i])
        {
            std::cout << "Stencil CPU and GPU are NOT equivalent!" << std::endl;
            std::cout << "Index: " << i << std::endl;
            std::cout << "CPU: " << cpu_result[i] << std::endl;
            std::cout << "GPU: " << gpu_results[i] << std::endl;
            return EXIT_FAILURE;
        }
    }
    std::cout << "Stencil CPU and GPU are equivalent!" << std::endl;
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
