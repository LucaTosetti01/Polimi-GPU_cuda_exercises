# GPU CUDA Exercises

Lab exercises and exams for the "GPU & Heterogeneous systems" course 2023/2024 @ Polimi. \
Prof: Antonio Miele Rosario.

This repository contains several CUDA exercise proposed by the professor during the course and several exams with my personal solutions, you can check them by browsing the branches.
Branch summary:


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