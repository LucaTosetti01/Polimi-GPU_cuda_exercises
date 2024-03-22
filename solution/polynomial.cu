/*
DESCRIPTION: This program takes in input N different integer vectors a, 
each one of M elements, and another vector b of M integer elements. The 
program computes the dot product of each vector a[i] and the vector b.
The program executes the dot products first on the CPU and then on the 
GPU. In a second GPU-accelerated version of the dot product function, 
constant memory is used to store b.
DO NOTE: The program stores the N input vectors in a MxN matrix where 
each column represents a vector (by following a "Struct of Arrays" 
organization of the data); this allow an optimization of the global 
memory load accesses when executing on the GPU.

GOAL: use constant memory to improve performance 
COMPILE: nvcc polynomial.cu -o polynomial
PROFILE: ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./polynomial
*/

#include<stdio.h>
#include<cuda_runtime.h>
#include <sys/time.h>

#define N (1<<14)
#define M 32
#define MAXVAL 10
#define BLOCKDIM 64

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

__constant__ int coeffs[M];

inline double milliseconds();
__global__ void dot_product(int* data, int* r, int* c, int dim, int val);
__global__ void dot_product2(int* data, int* r, int dim, int nc);
void dot_product_cpu(int* data, int* r, int* c, int dim, int nc);

inline double milliseconds(){
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec * 1000 + (double)tp.tv_usec * 0.001);
}

//compute N dot products on CPU
void dot_product_cpu(int* data, int* r, int* c, int dim, int nc) {
  int i, s, j;
  for(i=0; i<dim; i++){
    for(s=0, j=0; j<nc; j++)
      s += data[j*dim+i]*c[j];
    r[i] = s;
  }
}

//compute N dot products on GPU
__global__ void dot_product(int* data, int* r, int* c, int dim, int nc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int s, j;
  if(i<dim)
    for(s=0, j=0; j<nc; j++)
      s += data[j*dim+i]*c[j];
  r[i] = s;
}

//compute N dot products on GPU by using constant memory
__global__ void dot_product2(int* data, int* r, int dim, int nc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int s, j;
  if(i<dim)
    for(s=0, j=0; j<nc; j++)
      s += data[j*dim+i]*coeffs[j];
  r[i] = s;
}

int main(){    
  int h_data[M][N], h_r[N], h_r_cpu[N], h_c[M];
  int *d_data, *d_r, *d_c;
  int i, j, ok;
  double cpu_start, cpu_end, cpu_exectime;

  cudaEvent_t gpu_start1, gpu_end1;
  cudaEvent_t gpu_start2, gpu_end2;
  float gpu_exectime1;
  float gpu_exectime2;
 
  srand(0);
    
  /*initialize CUDA events*/
  CHECK(cudaEventCreate(&gpu_start1));
  CHECK(cudaEventCreate(&gpu_end1));
  CHECK(cudaEventCreate(&gpu_start2));
  CHECK(cudaEventCreate(&gpu_end2));

  /*initialize vectors*/  
  for(i=0; i<M; i++)
    for(j=0; j<N; j++)
      h_data[i][j] = rand()%MAXVAL;
  for(j=0; j<M; j++)
    h_c[j]=j;
  
  /*call CPU function*/
  cpu_start = milliseconds();
  dot_product_cpu(&h_data[0][0], h_r_cpu, h_c, N, M);
  cpu_end = milliseconds();
  cpu_exectime = cpu_end - cpu_start;

  /*allocate memory on the GPU*/
  CHECK(cudaMalloc(&d_data, M*N*sizeof(int)));
  CHECK(cudaMalloc(&d_r, N*sizeof(int)));
  CHECK(cudaMalloc(&d_c, M*sizeof(int)));

  /*transmit data to GPU*/
  CHECK(cudaMemcpy(d_data, &h_data[0][0], M*N*sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_c, h_c, M*sizeof(int), cudaMemcpyHostToDevice));

  /*invoke the kernel on the GPU*/
  dim3 blocksPerGrid1((N+BLOCKDIM-1)/BLOCKDIM, 1, 1);
  dim3 threadsPerBlock1(BLOCKDIM, 1, 1);
  CHECK(cudaEventRecord(gpu_start1));
  dot_product<<<blocksPerGrid1, threadsPerBlock1>>>(d_data, d_r, d_c, N, M);
  //CHECK_KERNELCALL();
  CHECK(cudaEventRecord(gpu_end1));
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventElapsedTime(&gpu_exectime1, gpu_start1, gpu_end1));
        
  /*transmit data from the GPU*/
  CHECK(cudaMemcpy(h_r, d_r, N*sizeof(int), cudaMemcpyDeviceToHost));

  /*compare results*/
  for(i=0, ok=1; i<N && ok; i++)
    if(h_r[i] != h_r_cpu[i]){
      printf("Error!\n");
      ok = 0;
    }
  if(ok)
    printf("Success!\n");
    
  /*transmit data to GPU*/
  CHECK(cudaMemcpyToSymbol(coeffs, h_c, M*sizeof(int)));
  CHECK(cudaMemcpy(d_data, &h_data[0][0], M*N*sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_c, h_c, M*sizeof(int), cudaMemcpyHostToDevice));

  /*invoke the kernel on the GPU*/
  dim3 blocksPerGrid2((N+BLOCKDIM-1)/BLOCKDIM, 1, 1);
  dim3 threadsPerBlock2(BLOCKDIM, 1, 1);
  CHECK(cudaEventRecord(gpu_start2));
  dot_product2<<<blocksPerGrid2, threadsPerBlock2>>>(d_data, d_r, N, M);
  //CHECK_KERNELCALL();
  CHECK(cudaEventRecord(gpu_end2));
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventElapsedTime(&gpu_exectime2, gpu_start2, gpu_end2));
        
  /*transmit data from the GPU*/
  CHECK(cudaMemcpy(h_r, d_r, N*sizeof(int), cudaMemcpyDeviceToHost));

  /*compare results*/
  for(i=0, ok=1; i<N && ok; i++)
    if(h_r[i] != h_r_cpu[i]){
      printf("Error!\n");
      ok = 0;
    }
  if(ok)
    printf("Success!\n");
    
  /*report ezxecution time*/  
  printf("Execution times\n- CPU: %f\n- GPU: %f\n- GPU: %f\n", cpu_exectime, gpu_exectime1, gpu_exectime2);

  /*free memory*/
  CHECK(cudaEventDestroy(gpu_start1));
  CHECK(cudaEventDestroy(gpu_end1));
  CHECK(cudaEventDestroy(gpu_start2));
  CHECK(cudaEventDestroy(gpu_end2));
  CHECK(cudaFree(d_data));
  CHECK(cudaFree(d_r));
  CHECK(cudaFree(d_c));
  
  return 0;
}



