/*
DESCRIPTION: it is a program running a dummy kernel copying an input 
vector in an output one.

GOAL: profile global memory accesses 
COMPILE: nvcc analyze_memory_accesses.cu -o analyze_memory_accesses
PROFILE: ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./analyze_memory_accesses

Change N to 1, 16, 48, 64 and analyze the number of transactions 
to the global memory and the global load/store efficiency
similarly (with N=32) replace in the kernel code "b[i] = a[i];" with
"b[i] = a[(i*2)%BLOCKDIM];" or "b[i] = a[0];" or "b[i] = a[i+1];"
change also the index of b.
Finally, analyze also different data types changing the typedef instruction.
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


#define N 32
#define DATASIZE N*2
#define BLOCKDIM N

typedef char mytype;

__global__ void foo(mytype* a, mytype* b);

__global__ void foo(mytype* a, mytype* b){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  b[i] = a[i]; 
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

  /*free memory*/
  CHECK(cudaFree(d_va));
  CHECK(cudaFree(d_vb));
  
  return 0;
}
