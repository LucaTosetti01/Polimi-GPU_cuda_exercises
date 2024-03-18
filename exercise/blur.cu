#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CHANNELS 3
#define OUT_FN_CPU "output.pgm"
#define BLURDIM 10

inline double milliseconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec * 1000 + (double)tp.tv_usec * 0.001);
}

enum IMAGE_TYPE
{
    ppm,
    pgm
};

int save_ppm_image(const char *filename, unsigned char *image, unsigned int width, unsigned int height);
int save_pgm_image(const char *filename, unsigned char *image, unsigned int width, unsigned int height);
int load_image(const char *filename, unsigned char **image, unsigned int *width, unsigned int *height, IMAGE_TYPE imgType);
void rgb2gray(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height);
void blur(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height);

__global__ void rgb2grayKernel(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned char redValue, greenValue, blueValue, grayValue;

    if (i < height && j < width)
    {
        redValue = input[(i * width + j) * 3];
        greenValue = input[(i * width + j) * 3 + 1];
        blueValue = input[(i * width + j) * 3 + 2];
        grayValue = (unsigned char)(0.299 * redValue + 0.587 * greenValue + 0.114 * blueValue);
        output[i * width + j] = grayValue;
    }
}

__global__ void blurKernel(unsigned char *input, unsigned char *output, unsigned int width, unsigned int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int h, k, sum = 0, count = 0;
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

int main(int argc, char *argv[])
{
    char *inputfile;
    unsigned int height, width;
    unsigned char *input, *gray, *output;
    unsigned char *d_input, *d_gray, *d_output;
    int nPixels;
    int err;

    double cpu_start, cpu_end, cpu_exectime;
    cudaEvent_t gpu_start;
    cudaEvent_t gpu_end;
    float gpu_exectime;

    // read arguments
    if (argc != 4)
    {
        printf("Please specify ppm input file name\n");
        return 0;
    }
    inputfile = argv[1];
    int blockDimX = atoi(argv[2]);
    int blockDimY = atoi(argv[3]);

    // load input image
    err = load_image(inputfile, &input, &width, &height, IMAGE_TYPE::ppm);

    if (err)
        return 1;
    nPixels = width * height;
    printf("prova %d\n",nPixels);
    // allocate memory for gray image
    gray = (unsigned char *)malloc(sizeof(unsigned char) * nPixels);
    if (!gray)
    {
        printf("Error with malloc\n");
        free(input);
        return 1;
    }
    printf("prova\n");
    // allocate memory for output image
    output = (unsigned char *)malloc(sizeof(unsigned char) * nPixels);
    if (!output)
    {
        printf("Error with malloc\n");
        free(gray);
        free(input);
        return 1;
    }
    printf("prova2\n");

    // process image
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
        free(gray);
        free(output);
        return 1;
    }

    cudaMalloc(&d_input, sizeof(unsigned char) * nPixels * CHANNELS);
    cudaMalloc(&d_gray, sizeof(unsigned char) * nPixels);
    cudaMalloc(&d_output, sizeof(unsigned char) * nPixels);
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_end);

    cudaMemcpy(d_input, input, sizeof(unsigned char) * nPixels * CHANNELS, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(ceil((float)width / blockDimX), ceil((float)height / blockDimY));
    dim3 threadsPerBlock(blockDimX, blockDimY);

    int dev;
    cudaDeviceProp deviceProp;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&deviceProp, dev);

    if (threadsPerBlock.x <= 0 || threadsPerBlock.x > deviceProp.maxThreadsDim[0] ||
        threadsPerBlock.y <= 0 || threadsPerBlock.y > deviceProp.maxThreadsDim[1])
    {
        printf("Violeted the minimum or maximum size of the dimension of a block(0;%d] - (0;%d]",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1]);
        free(input);
        free(output);
        free(gray);
        cudaDeviceReset();
        return 1;
    }

    if (threadsPerBlock.x * threadsPerBlock.y > deviceProp.maxThreadsPerBlock)
    {
        printf("Violeted the maximum number of threads per block(0;%d]",
               deviceProp.maxThreadsPerBlock);
        free(input);
        free(output);
        free(gray);
        cudaDeviceReset();
        return 1;
    }

    cudaEventRecord(gpu_start);
    rgb2grayKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_gray, width, height);
    blurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_gray, d_output, width, height);
    cudaEventRecord(gpu_end);
    cudaEventSynchronize(gpu_end);
    cudaEventElapsedTime(&gpu_exectime, gpu_start, gpu_end);

    cudaMemcpy(output, d_output, sizeof(unsigned char) * nPixels, cudaMemcpyDeviceToHost);

    // save output image
    err = save_pgm_image(OUT_FN_CPU, output, width, height);
    if (err)
    {
        free(input);
        free(gray);
        free(output);
        cudaDeviceReset();
        return 1;
    }

    printf("Host's execution time: %f\n", cpu_exectime);
    printf("Kernel function's execution time: %f\n", gpu_exectime);
    bool ifDeviceFaster = gpu_exectime < cpu_exectime ? 1 : 0;
    printf("%s was %f milliseconds faster!\n", (ifDeviceFaster ? "Device" : "Host"), (ifDeviceFaster ? cpu_exectime - gpu_exectime : gpu_exectime - cpu_exectime));

    // cleanup
    free(input);
    free(gray);
    free(output);
    cudaDeviceReset();

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

int load_image(const char *filename, unsigned char **image, unsigned int *width, unsigned int *height, IMAGE_TYPE imgType)
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

    printf("prova %d", (imgType == IMAGE_TYPE::ppm) ? CHANNELS : 1);
    *image = (unsigned char *)malloc(sizeof(unsigned char) * (*width) * (*height) * (imgType == IMAGE_TYPE::ppm) ? CHANNELS : 1);
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
