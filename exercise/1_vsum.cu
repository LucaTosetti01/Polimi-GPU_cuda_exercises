/*
Vector addition.
* Version 1: the sum is performed by a function on the GPU
*/

#include <stdio.h>
#include <cuda_runtime.h>

/*Number of data to be processed contained in each vector*/
#define N (1 << 16) /*65536*/
/*Block size*/
#define BLOCKDIM 64

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
  cudaMalloc(&d_va, N * sizeof(int));
  cudaMalloc(&d_vb, N * sizeof(int));
  cudaMalloc(&d_vc, N * sizeof(int));

  /*Copying the host's vectors 'h_a' and 'h_b' into device's vectors 'd_va' and 'd_vb'*/
  cudaMemcpy(d_va, h_va, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vb, h_vb, N * sizeof(int), cudaMemcpyHostToDevice);

  /*Declaring structs that defines the number of blocks and thread within them*/
  dim3 blocksPerGrid((N + BLOCKDIM - 1) / BLOCKDIM, 1, 1);
  dim3 threadPerBlock(BLOCKDIM, 1, 1);
  /*Invoking kernel function executed on the Device*/
  vsumKernel<<<blocksPerGrid, threadPerBlock>>>(d_va, d_vb, d_vc, N);

  /*Copying the results of the kernel function from the Device's vector 'd_vc' to host's vector 'h_vc'*/
  cudaMemcpy(h_vc, d_vc, N * sizeof(int), cudaMemcpyDeviceToHost);

  /*Freeing precedently allocated memory in the Device*/
  cudaFree(d_va);
  cudaFree(d_vb);
  cudaFree(d_vc);

  /*we don't print the results...*/
  printf("Done!\n");

  return 0;
}
