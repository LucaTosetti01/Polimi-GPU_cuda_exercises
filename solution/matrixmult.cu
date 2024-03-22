/*
DESCRIPTION: this is the matrix multiplication example shown in the class
on CUDA  memory model. The code includes 3 alternative kernel versions: 
1) CPU implementation, 
2) naive GPU implementation, and 
3) tiled GPU implementation

GOAL: profile how shared memory helps improving performance by reducing 
global memory traffic
COMPILE: nvcc matrixmult.cu -o matrixmult
EXECUTE: ./matrixmult 1024 1024 1024 1024
PROFILE: ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./matrixmult 32 32 32 32

Change the naive kernel implementation by removing sum variable and 
directly accumulating partial results on P[i * numNColumns + j]. Check
how the number of global memory stores increases considerably.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define BLOCKDIM 32
#define TILE_WIDTH BLOCKDIM
#define MAXVAL 8

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

inline double milliseconds();
void printM(int *M, int numMRows, int numMColumns);
void matrixmult(int *M, int *N, int *P, int numMRows, int numMColumns, int numNColumns);
__global__ void sgemm(int *M, int *N, int *P, int numMRows, int numMColumns, int numNColumns);
__global__ void sgemm_sharedmem(int *M, int *N, int *P, int numMRows, int numMColumns, int numNColumns);


//display a matrix on the screen
void printM(int *M, int numMRows, int numMColumns) {
  int i, j;
  for(i=0; i<numMRows; i++){
    for(j=0; j<numMColumns; j++)
      printf("%3d ", M[i * numMColumns + j]);
    printf("\n");
  }
  printf("\n");
}

//compute matrix multiplication on CPU
void matrixmult(int *M, int *N, int *P, int numMRows, int numMColumns, int numNColumns) {
  int i, j, k;
  for(i=0; i<numMRows; i++)
    for(j=0; j<numNColumns; j++)
      for(k=0, P[i * numMColumns + j]=0; k<numMColumns; k++)
        P[i * numMColumns + j] += M[i * numMColumns + k] * N[k * numNColumns + j];  
}


//compute matrix multiplication on CPU
__global__ void basic_matrixmult(int *M, int *N, int *P, int numMRows, int numMColumns, int numNColumns) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numMRows && j < numNColumns) {
    int sum = 0;
    for (int k = 0; k < numMColumns; k++)
      sum += M[i * numMColumns + k] * N[k * numNColumns + j];
    P[i * numNColumns + j] = sum;
  }
}

//compute tiled matrix multiplication on CPU
__global__ void tiled_matrixmult(int *M, int *N, int *P, int numMRows, int numMColumns, int numNColumns) {
  __shared__ int ds_M[TILE_WIDTH][TILE_WIDTH];
  __shared__ int ds_N[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x, 
      by = blockIdx.y, 
      tx = threadIdx.x, 
      ty = threadIdx.y,
      i = by * TILE_WIDTH + ty, 
      j = bx * TILE_WIDTH + tx;
  int Pvalue = 0;

  for (int m = 0; m < (numMColumns - 1) / TILE_WIDTH + 1; ++m) {
    if (i < numMRows && m * TILE_WIDTH + tx < numMColumns)
      ds_M[ty][tx] = M[i * numMColumns + m * TILE_WIDTH + tx];
    else
      ds_M[ty][tx] = 0;
    if (j < numNColumns && m * TILE_WIDTH + ty < numMColumns)
      ds_N[ty][tx] = N[(m * TILE_WIDTH + ty) * numNColumns + j];
    else
      ds_N[ty][tx] = 0;
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k){
      Pvalue += ds_M[ty][k] * ds_N[k][tx];
    }
    __syncthreads();
  }
  if (i < numMRows && j < numNColumns)
    P[i * numNColumns + j] = Pvalue;
}

int main(int argc, char **argv) {
  int *h_A; // The A matrix
  int *h_B; // The B matrix
  int *h_C; // The output C matrix
  int *h_C_cpu; // The output C matrix computed on the CPU
  int *d_A;
  int *d_B;
  int *d_C;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;
  int ok, i;
  
  //stuff to measure the time
  double cpu_start, cpu_end, cpu_exectime;
  cudaEvent_t gpu_start1;
  cudaEvent_t gpu_end1;
  cudaEvent_t gpu_start2;
  cudaEvent_t gpu_end2;
  float gpu_exectime1;
  float gpu_exectime2;

  //read arguments
  if(argc!=5){
    printf("Please specify sizes (#rows and #columns) of matrix A and B\n");
    return 0;
  }
  numARows=atoi(argv[1]);
  numAColumns=atoi(argv[2]);
  numBRows=atoi(argv[3]);
  numBColumns=atoi(argv[4]);
  
  if(numAColumns!=numBRows){
    printf("# colums of A is different from the number of rows of B\n");
    return 0;
  }

  //compute output matrix size
  numCRows=numARows;
  numCColumns=numBColumns;

  //allocate memory for the three matrices
  h_A = (int*) malloc(sizeof(int) * numARows*numAColumns);
  if(!h_A){
  	printf("Error: malloc failed\n");
  	return 1;
  }
  h_B = (int*) malloc(sizeof(int) * numBRows*numBColumns);
  if(!h_B){
  	printf("Error: malloc failed\n");
  	free(h_A);
  	return 1;
  }
  h_C = (int*) malloc(sizeof(int) * numCRows*numCColumns);
  if(!h_C){
  	printf("Error: malloc failed\n");
  	free(h_A);
  	free(h_B);
  	return 1;
  }
  h_C_cpu = (int*) malloc(sizeof(int) * numCRows*numCColumns);
  if(!h_C_cpu){
  	printf("Error: malloc failed\n");
  	free(h_A);
  	free(h_B);
  	free(h_C);
  	return 1;
  }
  //initialize input matrices
  srand(0);
  for(i=0; i<numARows*numAColumns; i++)
    h_A[i] = rand()%MAXVAL;
  for(i=0; i<numBRows*numBColumns; i++)
    h_B[i] = rand()%MAXVAL;

  //execute on CPU  
  cpu_start = milliseconds();
  matrixmult(h_A, h_B, h_C_cpu, numARows, numAColumns, numBColumns);
  cpu_end = milliseconds();
  cpu_exectime = cpu_end - cpu_start;
  
  //allocate device memory and transfer data
  CHECK(cudaMalloc(&d_A, numARows * numAColumns * sizeof(int)));
  CHECK(cudaMalloc(&d_B, numBRows * numBColumns * sizeof(int)));
  CHECK(cudaMalloc(&d_C, numCRows * numCColumns * sizeof(int)));
  CHECK(cudaEventCreate(&gpu_start1));
  CHECK(cudaEventCreate(&gpu_end1));
  CHECK(cudaEventCreate(&gpu_start2));
  CHECK(cudaEventCreate(&gpu_end2));
  
  CHECK(cudaMemcpy(d_A, h_A, numARows * numAColumns * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_B, h_B, numBRows * numBColumns * sizeof(int), cudaMemcpyHostToDevice));

  //execute on GPU 1
  dim3 blockDim(BLOCKDIM, BLOCKDIM);
  dim3 gridDim(ceil(((float)numBColumns) / blockDim.x), ceil(((float)numARows) / blockDim.y));

  CHECK(cudaEventRecord(gpu_start1));
  basic_matrixmult<<<gridDim, blockDim>>>(d_A, d_B, d_C, numARows, numAColumns, numBColumns);
  CHECK_KERNELCALL();
  CHECK(cudaEventRecord(gpu_end1));
  CHECK(cudaDeviceSynchronize());
  
  CHECK(cudaEventElapsedTime(&gpu_exectime1, gpu_start1, gpu_end1));

  //transfer results 1
  CHECK(cudaMemcpy(h_C, d_C, numCRows * numCColumns * sizeof(int), cudaMemcpyDeviceToHost));
  
  //check results
  for(i=0, ok=1; i<numCRows*numCColumns; i++)
    if(h_C[i]!=h_C_cpu[i]){
      ok=0;
      printf("%d %d %d\n", i, h_C[i], h_C_cpu[i]);
    }
  printf("Result: %s\n", ok?"OK":"NO");

  //execute on GPU 2
  CHECK(cudaEventRecord(gpu_start2));
  tiled_matrixmult<<<gridDim, blockDim>>>(d_A, d_B, d_C, numARows, numAColumns, numBColumns);
  CHECK_KERNELCALL();
  CHECK(cudaEventRecord(gpu_end2));
  CHECK(cudaDeviceSynchronize());
  
  CHECK(cudaEventElapsedTime(&gpu_exectime2, gpu_start2, gpu_end2));
  
  //transfer results 2
  CHECK(cudaMemcpy(h_C, d_C, numCRows * numCColumns * sizeof(int), cudaMemcpyDeviceToHost));
  
  //check results
  for(i=0, ok=1; i<numCRows*numCColumns; i++)
    if(h_C[i]!=h_C_cpu[i]){
      ok=0;
      printf("%d %d %d\n", i, h_C[i], h_C_cpu[i]);
    }
  printf("Result: %s\n", ok?"OK":"NO");

/*  printM(h_A, numMRows, numMColumns);
  printM(h_B, numNRows, numNColumns);
  printM(h_C, numPRows, numPColumns);
  printM(h_C_cpu, numPRows, numPColumns);*/
  
  printf("Execution times\n- CPU: %f\n- GPU: %f\n- GPU (shared mem): %f\n", cpu_exectime, gpu_exectime1, gpu_exectime2);

  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_C));
  
  CHECK(cudaEventDestroy(gpu_start1));
  CHECK(cudaEventDestroy(gpu_end1));
  CHECK(cudaEventDestroy(gpu_start2));
  CHECK(cudaEventDestroy(gpu_end2));

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_cpu);
  
  return 0;
}


inline double milliseconds(){
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec * 1000 + (double)tp.tv_usec * 0.001);
}
