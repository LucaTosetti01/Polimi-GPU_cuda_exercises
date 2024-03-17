/*
Vector addition.
* Version 2: the sum is performed by a function on the GPU with error handling
*/

#include <stdio.h>
#include <cuda_runtime.h>

/*Number of data to be processed contained in each vector*/
#define N (1 << 16) /*65536*/
/*Defining a block size that will cause an execption on Device side*/
#define BLOCKDIM -1

/*Macro used for checking possible errors on standard CUDA methods*/
#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

/*Macro used for checking possible errors on kernel functions*/
#define CHECK_KERNEL()                                                                    \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

/*Kernel function performing the addition and executed on the Device(GPU)*/
__global__ void vsumKernel(int *a, int *b, int *c, int dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim)
        c[i] = a[i] + b[i];
}

int main()
{
    /*Declaring the host vectors which will contains the data to be processed*/
    int h_va[N], h_vb[N], h_vc[N];
    /*Declaring the device vectors*/
    int *d_va, *d_vb, *d_vc;
    int i;

    /*initialize host vectors*/
    for (i = 0; i < N; i++)
    {
        h_va[i] = i;
        h_vb[i] = N - i;
    }

    /*Allocating space dynamically in the device mem. for the device's vectors*/
    CHECK(cudaMalloc(&d_va, N * sizeof(int)));
    CHECK(cudaMalloc(&d_vb, N * sizeof(int)));
    CHECK(cudaMalloc(&d_vc, N * sizeof(int)));

    /*Copying the host's vectors 'h_a' and 'h_b' into device's vectors 'd_va' and 'd_vb'*/
    CHECK(cudaMemcpy(d_va, h_va, sizeof(int) * N, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_vb, h_vb, N * sizeof(int), cudaMemcpyHostToDevice));

    /*Declaring structs that defines the number of blocks and thread within them*/
    dim3 blocksPerGrid((N + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
    dim3 threadPerBlock(BLOCKDIM, 1, 1);
    /*Invoking kernel function executed on the Device*/
    vsumKernel<<<blocksPerGrid, threadPerBlock>>>(d_va, d_vb, d_vc, N);
    CHECK_KERNEL();

    /*Copying the results of the kernel function from the Device's vector 'd_vc' to host's vector 'h_vc'*/
    CHECK(cudaMemcpy(h_vc, d_vc, N * sizeof(int), cudaMemcpyDeviceToHost));

    /*Freeing precedently allocated memory in the Device*/
    CHECK(cudaFree(d_va));
    CHECK(cudaFree(d_vb));
    CHECK(cudaFree(d_vc));
    /*we don't print the results...*/
    printf("Done!\n");

    return 0;
}
