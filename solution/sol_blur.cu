#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define CHANNELS 3
#define OUT_FN_CPU "output_cpu.pgm"
#define OUT_FN_GPU "output_gpu.pgm"
#define BLURDIM 10

inline double milliseconds();
int save_ppm_image(const char *filename, unsigned char *image, unsigned int width, unsigned int height);
int save_pgm_image(const char *filename, unsigned char *image, unsigned int width, unsigned int height);
int load_ppm_image(const char *filename, unsigned char **image, unsigned int *width, unsigned int *height);
int load_pgm_image(const char *filename, unsigned char **image, unsigned int *width, unsigned int *height);
void rgb2gray(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height);
void blur(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height);

void rgb2gray(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height)
{
  int i, j;
  unsigned char redValue, greenValue, blueValue, grayValue;
  // loop on all pixels and convert from RGB to gray scale
  for (i = 0; i < height; i++)
  {
    for (j = 0; j < width; j++)
    {
      redValue = input[(i * width + j) * 3];
      greenValue = input[(i * width + j) * 3 + 1];
      blueValue = input[(i * width + j) * 3 + 2];
      grayValue = (unsigned char)(0.299 * redValue + 0.587 * greenValue + 0.114 * blueValue);
      output[i * width + j] = grayValue;
    }
  }
}

void blur(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height)
{
  int i, j, h, k, sum, count;
  // loop on all pixels and to compute the mean value of the intensity together with the 8 neighbor pixels
  for (i = 0; i < height; i++)
  {
    for (j = 0; j < width; j++)
    {
      count = 0;
      sum = 0;
      for (h = -BLURDIM; h <= BLURDIM; h++)
        for (k = -BLURDIM; k <= BLURDIM; k++)
          if (i + h >= 0 && i + h < height && j + k >= 0 && j + k < width)
          {
            count++;
            sum = sum + input[(i + h) * width + (j + k)];
          }
      output[i * width + j] = (float)sum / count;
    }
  }
}

__global__ void rgb2grayGPU(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height)
{
  unsigned char redValue, greenValue, blueValue, grayValue;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < width && i < height)
  {
    redValue = input[(i * width + j) * 3];
    greenValue = input[(i * width + j) * 3 + 1];
    blueValue = input[(i * width + j) * 3 + 2];
    grayValue = (unsigned char)(0.299 * redValue + 0.587 * greenValue + 0.114 * blueValue);
    output[i * width + j] = grayValue;
  }
}

__global__ void blurGPU(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height)
{
  int sum, count, h, k;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  count = 0;
  sum = 0;
  if (j < width && i < height)
  {
    for (h = -BLURDIM; h <= BLURDIM; h++)
      for (k = -BLURDIM; k <= BLURDIM; k++)
        if (i + h >= 0 && i + h < height && j + k >= 0 && j + k < width)
        {
          count++;
          sum = sum + input[(i + h) * width + (j + k)];
        }
    output[i * width + j] = (float)sum / count;
  }
}

int main(int argc, char *argv[])
{
  char *inputfile;
  unsigned int height, width;
  unsigned char *input, *gray, *output;
  unsigned char *input_d, *gray_d, *output_d;
  int nPixels;
  int err;

  double cpu_start, cpu_end, cpu_exectime;
  cudaEvent_t gpu_start;
  cudaEvent_t gpu_end;
  float gpu_exectime;

  // read arguments
  if (argc != 4)
  {
    printf("Please specify ppm input file name and 2 integer values for X and Y block sizes\n");
    return 0;
  }
  inputfile = argv[1];
  int blockdim_x = atoi(argv[2]);
  int blockdim_y = atoi(argv[3]);

  // load input image
  err = load_ppm_image(inputfile, &input, &width, &height);
  if (err)
    return 1;
  nPixels = width * height;

  // allocate memory for gray image
  gray = (unsigned char *)malloc(sizeof(unsigned char) * nPixels);
  if (!gray)
  {
    printf("Error with malloc\n");
    free(input);
    return 1;
  }

  // allocate memory for output image
  output = (unsigned char *)malloc(sizeof(unsigned char) * nPixels);
  if (!output)
  {
    printf("Error with malloc\n");
    free(input);
    return 1;
  }

  // process image on CPU
  cpu_start = milliseconds();
  rgb2gray(input, gray, width, height);
  blur(gray, output, width, height);
  cpu_end = milliseconds();
  cpu_exectime = cpu_end - cpu_start;

  // save output image
  err = save_pgm_image(OUT_FN_CPU, output, width, height);
  if (err)
  {
    free(input);
    free(output);
    return 1;
  }

  // allocate memory on the GPU
  cudaMalloc(&input_d, sizeof(unsigned char) * nPixels * CHANNELS);
  cudaMalloc(&gray_d, sizeof(unsigned char) * nPixels);
  cudaMalloc(&output_d, sizeof(unsigned char) * nPixels);
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_end);

  // transmit data to GPU
  cudaMemcpy(input_d, input, sizeof(unsigned char) * nPixels * CHANNELS, cudaMemcpyHostToDevice);

  // invoke the kernel on the GPU
  dim3 blocksPerGrid((width + blockdim_x - 1) / blockdim_x, (height + blockdim_y - 1) / blockdim_y, 1);
  dim3 threadsPerBlock(blockdim_x, blockdim_y, 1);

  int dev;
  cudaDeviceProp deviceProp;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&deviceProp, dev);

  if (threadsPerBlock.x <= 0 || threadsPerBlock.x > deviceProp.maxThreadsDim[0] ||
      threadsPerBlock.y <= 0 || threadsPerBlock.y > deviceProp.maxThreadsDim[1])
  {
    printf("Violated maximum sizes of a dimension of a block (0;%d] - (0:%d] - Specified values: %d %d\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           threadsPerBlock.x, threadsPerBlock.y);
    free(input);
    free(output);
    cudaDeviceReset();
    return 1;
  }
  if (threadsPerBlock.x * threadsPerBlock.y > deviceProp.maxThreadsPerBlock)
  {
    printf("Violated maximum number of threads per block (%d) - Specified value: %d\n",
           deviceProp.maxThreadsPerBlock, threadsPerBlock.x * threadsPerBlock.y);
    free(input);
    free(output);
    cudaDeviceReset();
    return 1;
  }

  cudaEventRecord(gpu_start);
  rgb2grayGPU<<<blocksPerGrid, threadsPerBlock>>>(input_d, gray_d, width, height);
  blurGPU<<<blocksPerGrid, threadsPerBlock>>>(gray_d, output_d, width, height);
  cudaEventRecord(gpu_end);
  cudaEventSynchronize(gpu_end);
  // cudaDeviceSynchronize();
  cudaEventElapsedTime(&gpu_exectime, gpu_start, gpu_end);

  // transmit data from the GPU
  cudaMemcpy(output, output_d, nPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // save output image
  err = save_pgm_image(OUT_FN_GPU, output, width, height);
  if (err)
  {
    free(input);
    free(output);
    cudaDeviceReset();
    return 1;
  }

  printf("Execution times\n- CPU: %f\n- GPU: %f\n", cpu_exectime, gpu_exectime);

  // cleanup
  free(input);
  free(gray);
  free(output);
  cudaFree(input_d);
  cudaFree(gray_d);
  cudaFree(output_d);
  cudaEventDestroy(gpu_start);
  cudaEventDestroy(gpu_end);

  return 0;
}

int save_ppm_image(const char *filename, unsigned char *image, unsigned int width, unsigned int height)
{
  FILE *f; // output file handle

  // open the output file and write header info for PPM filetype
  f = fopen(filename, "wb");
  if (f == NULL)
  {
    fprintf(stderr, "Error opening 'output.ppm' output file\n");
    return -1;
  }
  fprintf(f, "P6\n");
  fprintf(f, "%d %d\n%d\n", width, height, 255);
  fwrite(image, sizeof(unsigned char), height * width * CHANNELS, f);
  fclose(f);
  return 0;
}

int save_pgm_image(const char *filename, unsigned char *image, unsigned int width, unsigned int height)
{
  FILE *f; // output file handle

  // open the output file and write header info for PPM filetype
  f = fopen(filename, "wb");
  if (f == NULL)
  {
    fprintf(stderr, "Error opening 'output.ppm' output file\n");
    return -1;
  }
  fprintf(f, "P5\n");
  fprintf(f, "%d %d\n%d\n", width, height, 255);
  fwrite(image, sizeof(unsigned char), height * width, f);
  fclose(f);
  return 0;
}

int load_ppm_image(const char *filename, unsigned char **image, unsigned int *width, unsigned int *height)
{
  FILE *f; // input file handle
  char temp[256];
  unsigned int s;

  // open the input file and write header info for PPM filetype
  f = fopen(filename, "rb");
  if (f == NULL)
  {
    fprintf(stderr, "Error opening '%s' input file\n", filename);
    return -1;
  }
  fscanf(f, "%s\n", temp);
  fscanf(f, "%d %d\n", width, height);
  fscanf(f, "%d\n", &s);

  *image = (unsigned char *)malloc(sizeof(unsigned char) * (*width) * (*height) * CHANNELS);
  if (*image)
    fread(*image, sizeof(unsigned char), (*width) * (*height) * CHANNELS, f);
  else
  {
    printf("Error with malloc\n");
    return -1;
  }

  fclose(f);
  return 0;
}

int load_pgm_image(const char *filename, unsigned char **image, unsigned int *width, unsigned int *height)
{
  FILE *f; // input file handle
  char temp[256];
  unsigned int s;

  // open the input file and write header info for PPM filetype
  f = fopen(filename, "rb");
  if (f == NULL)
  {
    fprintf(stderr, "Error opening '%s' input file\n", filename);
    return -1;
  }
  fscanf(f, "%s\n", temp);
  fscanf(f, "%d %d\n", width, height);
  fscanf(f, "%d\n", &s);

  *image = (unsigned char *)malloc(sizeof(unsigned char) * (*width) * (*height));
  if (*image)
    fread(*image, sizeof(unsigned char), (*width) * (*height), f);
  else
  {
    printf("Error with malloc\n");
    return -1;
  }

  fclose(f);
  return 0;
}

inline double milliseconds()
{
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec * 1000 + (double)tp.tv_usec * 0.001);
}
