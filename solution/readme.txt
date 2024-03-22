analyze_memory_accesses.cu

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

----

analyze_divergence.cu

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

----

analyze_sharedmem.cu

DESCRIPTION: this program has been taken from the examples of "professional 
cuda c programming" textbook and slightly simplified. The code includes 
various kernels trasposing square thread coordinates of a CUDA grid into a 
global memory array. The various kernels in the code perform reads and writes 
with different ordering; padding is also used to reduce bank access conflicts.

GOAL: profile shared memory usage
COMPILE: nvcc analyze_sharedmem.cu -o analyze_sharedmem
PROFILE: ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct,smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum ./analyze_sharedmem

----

matrixmult.cu

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

----

rgb2gray

DESCRIPTION: this set of programs present various implementations of the RGB 
to gray scale conversion:
0_rgb2gray.cu - it is the basic implementation of the kernel
1_rgb2gray.cu - double constant values have been replaced with float ones 
(0.299f, 0.587f, 0.114f)
2_rgb2gray.cu - the bitmap image has been reshaped to separate the three 
channels (thus using a struct of arrays organization of the data)

GOAL: analyze how the adopted data types may impact on the performance
COMPILE: nvcc 0_rgb2gray.cu -o 0_rgb2gray; nvcc 1_rgb2gray.cu -o 1_rgb2gray; nvcc 2_rgb2gray.cu -o 2_rgb2gray
EXECUTE: ./0_rgb2gray input.ppm 32 32 ; ./1_rgb2gray input.ppm 32 32 ; ./2_rgb2gray input.ppm 32 32
PROFILE: 
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./0_rgb2gray input.ppm 32 32
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./1_rgb2gray input.ppm 32 32
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct ./2_rgb2gray input.ppm 32 32

----

polynomial.cu

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
