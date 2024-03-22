/*
DESCRIPTION: it is a program running a dummy kernel either computing 
the sum of all elements in the input vector or assigning 0 to the output. 
Based on the id, each thread will execute one of the two above elaborations.

GOAL: profile branch divergences
COMPILE: nvcc analyze_divergence.cu -o analyze_divergence
PROFILE: ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct ./analyze_divergence

Modify the if statement in the kernel code with each one of the following 
different instructions:

-> if(i%2) -> in each warp odd threads will take a direction while even 
              threads the other one.
-> if(i/32%2) -> all thread in each warp will take the same direction. 
                 The results are the same of the previous case but the 
                 output values are sorted in a different way! 
-> if(i==4) -> one thread in the first warp generates a divergence. 
               Other three warps have no divergence.
-> if(i==4|| i==40) -> two threads in two warps generate a divergence. 
                       Other two warps have no divergence.
*/

#include<stdio.h>
#include<cuda_runtime.h>

#define CHECK(call) \
{ \
  const cudaError_t err = call; \
  if (err != cudaSuccess) { \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} \

#define CHECK_KERNELCALL() \
{ \
  const cudaError_t err = cudaGetLastError();\
  if (err != cudaSuccess) {\
    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);\
  }\
}\


#define N 128
#define DATASIZE N
#define BLOCKDIM N

typedef int mytype;

__global__ void foo(mytype* a, mytype* b);

__global__ void foo(mytype* a, mytype* b){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int sum = 0;
  if(i%2)
    for(int j=0; j<N; j++)
      sum+=a[j];
  else
    sum = 0;

  b[i] = sum; 
}

int main(){
  mytype h_va[DATASIZE], h_vb[DATASIZE];
  mytype *d_va, *d_vb;
  int i;

  /*initialize vectors*/  
  for(i=0; i<DATASIZE; i++){
    h_va[i] = i;
  }
  
  /*allocate memory on the GPU*/
  CHECK(cudaMalloc(&d_va, DATASIZE*sizeof(mytype)));
  CHECK(cudaMalloc(&d_vb, DATASIZE*sizeof(mytype)));

  /*transmit data to GPU*/
  CHECK(cudaMemcpy(d_va, h_va, DATASIZE*sizeof(mytype), cudaMemcpyHostToDevice));

  /*invoke the kernel on the GPU*/
  dim3 blocksPerGrid((N+BLOCKDIM-1)/BLOCKDIM, 1, 1);
  dim3 threadsPerBlock(BLOCKDIM, 1, 1);
  foo<<<blocksPerGrid, threadsPerBlock>>>(d_va, d_vb);
  CHECK_KERNELCALL();
  
  /*transmit data from the GPU*/
  CHECK(cudaMemcpy(h_vb, d_vb, DATASIZE*sizeof(mytype), cudaMemcpyDeviceToHost));
  
  for(i=0; i<N; i++)
    printf("%d ", h_vb[i]);
  printf("\n");
  
  /*free memory*/
  CHECK(cudaFree(d_va));
  CHECK(cudaFree(d_vb));
  
  return 0;
}
