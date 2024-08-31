#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define MAX_LENGTH 50000000

#define CHAR_PER_BIN 6
#define ALPHABET_SIZE 26
#define NUM_BINS ((ALPHABET_SIZE - 1) / CHAR_PER_BIN + 1)

#define BLOCK_DIM 32
#define GRID_DIM 32

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

double get_time() // function to get the time of day in seconds
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void sequential_histogram(char *data, unsigned int *histogram, int length)
{
  for (int i = 0; i < length; i++)
  {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) // check if we have an alphabet char
      histogram[alphabet_position / 6]++;                 // we group the letters into blocks of 6
  }
}

__global__ void histogramGPU(char *data, unsigned int *histogram, int length)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  unsigned int accumulator = 0;
  int prevBinIdx = -1;

  __shared__ unsigned int private_histogram[NUM_BINS];
  // initializing shared memory
  for (unsigned int j = threadIdx.x; j < NUM_BINS; j += blockDim.x)
  {
    private_histogram[j] = 0u;
  }
  __syncthreads();

  for (int i = tid; i < length; i += stride)
  {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26)
    { // check if we have an alphabet char
      const int bin = alphabet_position / CHAR_PER_BIN;
      if (bin == prevBinIdx)
      {
        accumulator++;
      }
      else
      {
        if (accumulator > 0)
        {
          atomicAdd(&(private_histogram[bin]), accumulator);
        }
        prevBinIdx = bin;
        accumulator = 1;
      }
    }
  }
  if (accumulator > 0) {
    atomicAdd(&(private_histogram[prevBinIdx]), accumulator);
  }

  __syncthreads();

  for (unsigned int j = threadIdx.x; j < NUM_BINS; j += blockDim.x)
  {
    const int value = private_histogram[j];
    if (value > 0)
    {
      atomicAdd(&histogram[j], value);
    }
  }
}

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("Please provide a filename as an argument.\n");
    return 1;
  }

  const char *filename = argv[1];
  FILE *fp = fopen(filename, "read");

  // unsigned char text[MAX_LENGTH];
  char *text = (char *)malloc(sizeof(char) * MAX_LENGTH);
  char *d_text;
  size_t len = 0;
  size_t read;
  unsigned int histogram[NUM_BINS] = {0};
  unsigned int histogramGPU_res[NUM_BINS] = {0};
  unsigned int *d_histogram;

  if (fp == NULL)
    exit(EXIT_FAILURE);

  while ((read = getline(&text, &len, fp)) != -1)
  {
    printf("Retrieved line of length %ld:\n", read);
  }

  fclose(fp);

  sequential_histogram(text, histogram, len);

  printf("a-f: %d, g-l: %d, m-r: %d, s-x: %d, y-z: %d\n",
         histogram[0],
         histogram[1],
         histogram[2],
         histogram[3],
         histogram[4]);

  cudaMalloc(&d_text, sizeof(char) * len);
  cudaMalloc(&d_histogram, sizeof(unsigned int) * NUM_BINS);

  cudaMemcpy(d_text, text, sizeof(char) * len, cudaMemcpyHostToDevice);
  // cudaMemcpy(d_histogram, histogramGPU_res, sizeof(int) * NUM_BINS, cudaMemcpyHostToDevice);

  dim3 blocksPerGrid((len + BLOCK_DIM - 1) / BLOCK_DIM, 1, 1);
  dim3 threadsPerBlock(BLOCK_DIM, 1, 1);

  histogramGPU<<<blocksPerGrid, threadsPerBlock>>>(d_text, d_histogram, len);
  cudaDeviceSynchronize();

  cudaMemcpy(histogramGPU_res, d_histogram, sizeof(unsigned int) * NUM_BINS, cudaMemcpyDeviceToHost);

  printf("a-f: %d, g-l: %d, m-r: %d, s-x: %d, y-z: %d\n",
         histogramGPU_res[0],
         histogramGPU_res[1],
         histogramGPU_res[2],
         histogramGPU_res[3],
         histogramGPU_res[4]);
  for (size_t i = 0; i < NUM_BINS; i++)
  {
    if (histogramGPU_res[i] != histogram[i])
    {
      printf("Results NOT correct...");
      printf("Error on GPU at index: %ld\n", i);
      cudaDeviceReset();
      exit(1);
    }
  }

  printf("Results are correct!!");
  cudaDeviceReset();
  free(text);

  return 1;
}