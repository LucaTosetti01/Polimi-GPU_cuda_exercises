/*
Vector addition.
* Version 3: the sum is performed by a function on the GPU with error handling, CPU and GPU execution time are compared
* by using system times for the CPU and events for the GPU (differently from the 3a_vsum exercise, the usage of
* cuda events let us obtain a more precise result in computing the time needed by the GPU to perform the computation.
* For this reason in this exercise's variant I chose to define a TINY_DATA_SIZE macro in order to get a situation
* in which the CPU computes the task faster than the GPU, otherwhise it would have been always the opposite)
*/

#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// Set TINY_DATA_SIZE to 1 if you want to make sure the CPU runs faster
#define TINY_DATA_SIZE 0
/*Number of data to be processed contained in each vector*/
#define N (TINY_DATA_SIZE ? (1 << 19) : (1 << 9)) /*HUGE_DATA_SIZE ? 524288 : 512 */
/*Block size*/
#define BLOCKDIM 64

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

/*Returns the current system time in milliseconds (declared as an inline function, so that the
  GCC compiler is able to make calls to that function faster, see: https://gcc.gnu.org/onlinedocs/gcc/Inline.html)*/
inline double milliseconds()
{
    struct timeval tp;
    struct timezone tzp;
    int success = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec * 1000 + (double)tp.tv_usec * 0.001);
}

/*CPU function performing vector addition c = a + b*/
void vsum(int *a, int *b, int *c, int dim)
{
    int i;
    for (i = 0; i < dim; i++)
        c[i] = a[i] + b[i];
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
    // printf("%d",N);
    /*Declaring the host vectors which will contains the data to be processed*/
    int h_va[N], h_vb[N], h_vc[N];
    /*Declaring the device vectors*/
    int *d_va, *d_vb, *d_vc;
    int i;
    double h_start_time, h_end_time, h_exec_time;
    /*Declaring the starting and ending Device's computation events*/
    cudaEvent_t start_event, end_event;
    float d_exec_time;

    /*initialize host vectors*/
    for (i = 0; i < N; i++)
    {
        h_va[i] = i;
        h_vb[i] = N - i;
    }
    /*Initializing cuda events*/
    CHECK(cudaEventCreate(&start_event));
    CHECK(cudaEventCreate(&end_event));

    /*call CPU function*/
    h_start_time = milliseconds();
    vsum(h_va, h_vb, h_vc, N);
    h_end_time = milliseconds();
    h_exec_time = h_end_time - h_start_time;

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

    CHECK(cudaEventRecord(start_event));
    /*Invoking kernel function executed on the Device*/
    vsumKernel<<<blocksPerGrid, threadPerBlock>>>(d_va, d_vb, d_vc, N);
    CHECK_KERNEL();

    /*Synchronizing the Host and Device's execution, otherwise the Host (CPU) will continue its execution
      while the Device is executing the kernel function*/
    CHECK(cudaEventRecord(end_event));
    CHECK(cudaDeviceSynchronize());
    /*Computing and printing the execution time of the kernel function*/
    CHECK(cudaEventElapsedTime(&d_exec_time, start_event, end_event));

    printf("Host's execution time: %f\n", h_exec_time);
    printf("Kernel function's execution time: %f\n", d_exec_time);
    bool ifDeviceFaster = d_exec_time < h_exec_time ? 1 : 0;
    printf("%s was %f milliseconds faster!\n", (ifDeviceFaster ? "Device" : "Host"), (ifDeviceFaster ? h_exec_time - d_exec_time : d_exec_time - h_exec_time));
    /*Copying the results of the kernel function from the Device's vector 'd_vc' to host's vector 'h_vc'*/
    CHECK(cudaMemcpy(h_vc, d_vc, N * sizeof(int), cudaMemcpyDeviceToHost));

    /*Freeing precedently allocated memory in the Device*/
    CHECK(cudaFree(d_va));
    CHECK(cudaFree(d_vb));
    CHECK(cudaFree(d_vc));
    /*Deleting the CUDA events*/
    CHECK(cudaEventDestroy(start_event));
    CHECK(cudaEventDestroy(end_event));
    /*we don't print the results...*/
    printf("Done!\n");

    return 0;
}
