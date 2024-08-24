#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

// #define DIM (512 + 256)
#define DIM 16
#define SIZE (DIM * DIM * DIM)
#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

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
    if (err != cudaSuccess)                                                         \
    {                                                                               \
      printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  }

#define CHECK_KERNELCALL()                                                          \
  {                                                                                 \
    const cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                         \
    {                                                                               \
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
  __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // Indexes for the "in" vector:
  // In order to upload a input_tile in the shared memory from
  // the input matrix, i need to take (for each dimension) the
  // dimension of an output tile of the output matrix and multiply
  // it by blockIdx, this allow us to identify a specific output tile
  // of the output matrix along a certain dimension/direction.
  // Afterwards i need to add the threadIdx in order to identify
  // a single cell of the tile along that dimension, and finally since
  // i have to upload an "input_tile" i need to upload also the "halo" cells
  // In this case this is done by subtracting 1 since che input tile is 1 cell
  // bigger in each direction w.r.t the output tile.
  int const unsigned inRow = blockIdx.x * OUT_TILE_DIM + tx - 1;
  int const unsigned inCol = blockIdx.y * OUT_TILE_DIM + ty - 1;
  int const unsigned inDepth = blockIdx.z * OUT_TILE_DIM + tz - 1;

  if (inRow < N &&
      inCol < N &&
      inDepth < N)
  {
    in_s[tx][ty][tz] = in(inRow, inCol, inDepth);
  }
  __syncthreads();
  if (inRow >= 1 && inRow < N - 1 &&
      inCol >= 1 && inCol < N - 1 &&
      inDepth >= 1 && inDepth < N - 1)
  {
    if (tx >= 1 && tx < IN_TILE_DIM - 1 &&
        ty >= 1 && ty < IN_TILE_DIM - 1 &&
        tz >= 1 && tz < IN_TILE_DIM - 1)
    {
      out(inRow, inCol, inDepth) = c0 * in_s[tx][ty][tz] + c1 * in_s[tx][ty][tz - 1] + c2 * in_s[tx][ty][tz + 1] + c3 * in_s[tx][ty - 1][tz] +
                                   c4 * in_s[tx][ty + 1][tz] + c5 * in_s[tx - 1][ty][tz] + c6 * in_s[tx + 1][ty][tz];
    }
  }
}

int main()
{
  static_assert(DIM / OUT_TILE_DIM, "The dimension should be divisible by the number of output block computer by each block");
  // Generate random input data
  std::vector<float> input_data(SIZE);
  for (int i = 0; i < SIZE; ++i)
  {
    input_data[i] = rand() % 10;
  }

  // Compute stencil on CPU
  std::vector<float> cpu_result(SIZE);
  stencil_cpu(input_data.data(), cpu_result.data(), DIM);

  // Allocate memory on GPU
  float *input_data_gpu, *output_data_gpu;
  CHECK(cudaMalloc(&input_data_gpu, SIZE * sizeof(float)));
  CHECK(cudaMalloc(&output_data_gpu, SIZE * sizeof(float)));
  CHECK(cudaMemcpy(input_data_gpu, input_data.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));

  // Configure GPU kernel launch
  dim3 threads_per_block(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
  dim3 blocks_per_grid((DIM + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                       (DIM + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                       (DIM + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

  // Launch GPU kernel
  stencil_gpu<<<blocks_per_grid, threads_per_block>>>(input_data_gpu, output_data_gpu, DIM);
  CHECK_KERNELCALL();
  CHECK(cudaDeviceSynchronize());

  // Copy result back to host
  std::vector<float> gpu_result(SIZE);
  CHECK(cudaMemcpy(gpu_result.data(), output_data_gpu, SIZE * sizeof(float), cudaMemcpyDeviceToHost));

  // Compare CPU and GPU results
  for (int i = 0; i < SIZE; ++i)
    if (cpu_result[i] != gpu_result[i])
    {
      std::cout << "Stencil CPU and GPU are NOT equivalent!" << std::endl;
      std::cout << "Index: " << i << std::endl;
      std::cout << "CPU: " << cpu_result[i] << std::endl;
      std::cout << "GPU: " << gpu_result[i] << std::endl;
      return EXIT_FAILURE;
    }

  std::cout << "Stencil CPU and GPU are equivalent!" << std::endl;

  // Free memory
  CHECK(cudaFree(input_data_gpu));
  CHECK(cudaFree(output_data_gpu));

  return EXIT_SUCCESS;
}
