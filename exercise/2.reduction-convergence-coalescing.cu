#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#define DIM 512
#define BLOCK_DIM 256

#define STRIDE_FACTOR 2
// CPU version of the reduction kernel
float reduce_cpu(const double *data, const int length)
{
    float sum = 0;
    for (int i = 0; i < length; i++)
    {
        sum += data[i];
    }
    return sum;
}

__global__ void reduce_gpu(double *__restrict__ input, double *__restrict__ output)
{
    const unsigned int i = threadIdx.x;

    // Apply the offset
    input += blockDim.x * blockIdx.x * STRIDE_FACTOR;
    output += blockIdx.x;

    for (unsigned int stride = blockDim.x; stride >= 1; stride /= STRIDE_FACTOR)
    {
        if (i < stride)
        {
            input[i] += input[i + stride];
        }

        __syncthreads();
    }

    // Write result for this block to global memory
    if (i == 0)
    {
        // You could have used only a single memory location and performed an atomicAdd
        *output = input[0];
    }
}

int main()
{
    std::vector<double> data(DIM);
    for (int i = 0; i < DIM; ++i)
        data[i] = static_cast<double>(rand()) / RAND_MAX; // Random value between 0 and 1 }

    // CPU version
    double sum_cpu = reduce_cpu(data.data(), data.size());

    std::cout << "Reduction CPU is " << sum_cpu << std::endl;

    double *d_data;
    double *d_output;

    cudaMalloc(&d_data, sizeof(double) * DIM);
    cudaMemcpy(d_data, data.data(), sizeof(double) * DIM, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_DIM, 1, 1);
    dim3 blocksPerGrid((DIM - 1) / (BLOCK_DIM * STRIDE_FACTOR) + 1, 1, 1);

    cudaMalloc(&d_output, sizeof(double) * blocksPerGrid.x);
    reduce_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_output);
    cudaDeviceSynchronize();

    double *gpu_results = (double *)malloc(sizeof(double) * blocksPerGrid.x);
    cudaMemcpy(gpu_results, d_output, sizeof(double) * blocksPerGrid.x, cudaMemcpyDeviceToHost);
    double sum_gpu = 0;

    for (int i = 0; i < blocksPerGrid.x; i++)
    {
        sum_gpu += gpu_results[i];
    }

    std::cout << "Reduction GPU is " << sum_gpu << std::endl;

    if (std::abs(sum_cpu - sum_gpu) > 1e-3)
    {
        std::cout << "Reduction CPU and GPU are NOT equivalent!" << std::endl;
        std::cout << "CPU: " << sum_cpu << std::endl;
        std::cout << "GPU: " << sum_gpu << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Reduction CPU and GPU are equivalent!" << std::endl;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output);
    return 0;
}
