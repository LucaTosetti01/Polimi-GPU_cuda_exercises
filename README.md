# GPU CUDA Exercises

Lab exercises and exams for the "GPU & Heterogeneous systems" course 2023/2024 @ Polimi. \
Prof: Antonio Miele Rosario.

This repository contains several CUDA exercise proposed by the professor during the course and several exams with my personal solutions, you can check them by browsing the branches. 

Grade obtained: 28/30

# Branch summary:


|Name|Description|Difficulty (Personally)|Main acceleration strategy(ies)||
|:----:|-----------|:-----------------------:|--------------------------|-|
|MAIN|Branch template for the creation of other branches|N.A.|N.A.|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/main)|
|VSUM|Accelerate the computation of the sum of 2 vectors, checking eventual error raised by CUDA commands and compare CPU and GPU time results|Very easy|Standard cuda parallelization with 1D grid|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/vsum)|
|BLUR|Accelerate the blurring of a ppm image|Easy|Standard cuda parallelization with 2D grid|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/blur)|
|SPARSE MATRICES|Accelerate the computation of systems of equations represented as sparse matrices|Easy / Medium|Data memory layout|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/sparse_matrices)|
|HISTOGRAM|Accelerate the computation of the letters' counting in a text using an histogram as result data structure|Medium|Coarsening / Privatization (With Global memory, Shared memory and registers)|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/histogram)|
|SCAN|Accelerate the "scan" (basically a foldr sums of the elements' array applied at every elements) process of an array of elements|Medium|Shared memory / Memory optimizations|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/scan)|
|EXAMS|Past course's exams with my and professor's solutions|Medium|Potentially all (but mainly Shared memory and Coarsening)|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/exams)|
|CONVOLUTION|Accelerate the convolution process between a matrix and a filter through several accelerating patterns|Medium/Hard|Constant memory / Tiling (Shared memory)|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/convolution)|
|REDUCTION|Accelerate the reduction on an array of elements|Medium / Hard|Coalesced accesses / Shared memory / Coarsening|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/reduction)|
|ELECTROSTATIC MAP|Accelerate the computation of an electrostatic potential map|Hard|Loop reorder optimization / Coarsening|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/electrostatic_map)|
|GRAPH TRAVERSAL|Accelerate the traversal of a graph through a BFS approach|Hard|Shared memory / Texture memory / Adapt kernel callings based on problem size|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/graph_traversal)|
|STENCIL|Accelerate the "stencil" operation (sum of the products between weight values and contiguous data elements surrounding a central data element)|Hard|Shared memory / Coarsening / Coarsening optimizations|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/stencil)|
|MERGE|Accelerate the merge of two arrays|Very hard|Shared memory|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/merge)|
|TOOLS PROFILING TESTS|Various exercises to be used in order to become pratical in using NVIDIA profiling tools (ncu)|N.A|N.A|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/tools_profiling_tests)|
|VARIOUS TESTS|Scratchpad branch used for testing various piece of codes used by the professor during the lectures|N.A|N.A|[Link](https://github.com/LucaTosetti01/GPU_cuda_exercises/tree/various_tests)|

# My dev environment

As my development environment i used Visual Studio Code installed on my personal Windows10 machine containing a wsl2 installation.

Specifications:
- Windows10: Version 22H2 (Build SO 19045.4170): It's necessary a Windows10's build version 19044+ with NVIDIA driver r545+ in order to access all CUDA tools(see [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#:~:text=Developer%20tools%20%2D%20Profilers%20%2D%20Volta%20and%20later%20(Using%20Windows%2010%20OS%20build%2019044%2B%20with%20driver%20r545%2B%20or%20using%20Windows%2011%20with%20driver%20r525%2B%20)) for more details)
- GPU Driver: 551.76
- Wsl2: Ubuntu22.04

# CUDA-Toolkit installation

A brief description of the steps that i took in order to make all work properly:
- I had already a wsl2 Ubuntu20.04 distro installed on my machine, but if you haven't just follow the simple NVIDIA "getting started" [here](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl-2).

> [!IMPORTANT]
> Ensure that you have the latest WSL kernel installed if you have already a WSL's distro installed.

- Verify if you have gcc installed in your WSL distro by using `gcc --version`. If you don't, run `sudo apt update` and then install it by using `sudo apt install build-essential`.
- At this point if you have a CUDA enabled GPU by running `nvidia-smi` in the WSL's console you should see your GPU informations.
- Follow the installation procedure stated on official CUDA-Toolkit download [page](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local) after selecting Linux -> x86_64 -> WSL-Ubuntu -> 2.0 -> deb(local).

> [!IMPORTANT]
> At this point you should be able to use CUDA toolkit compiler and profiler, however if like me nothing works try the subsequent steps

- Add at the end of the `.bashrc` file (that can be find at `/home/<yourUsername>`) the following lines: 
    - `export PATH="/usr/local/cuda-12.4/bin:$PATH"`
    - `export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"`
- Refresh the `.bashrc` file using `source .bashrc`
- Check whether cuda-toolkit was successfully installed by using:
    - `nvcc --version` : to see whether the compiler has been correctly installed or not
    - `ncu --version` : to see whether the profiler has been correctly installed or not

# Additional notes

When I created this repository, NVIDIA Nsight Systems did not support collecting kernels data in the timeline displayed when using WSL2 distributions, as documented [here](https://forums.developer.nvidia.com/t/nsys-is-not-collecting-kernel-data/244647). I have no clue whether the situation has changed or not







