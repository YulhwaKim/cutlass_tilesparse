# cutlass_tilesparse
CUDA templates for tile-sparse matrix multiplication based on [CUTLASS, NVIDA](https://github.com/NVIDIA/cutlass) [1].


# Introduction
Since Matrix Multiplication accounts for the largest part of the Neural Network computation, it is important to optimize Matrix Multiplication kernels for efficient Neural Network design.

In this project, we developed Tile-sparse Matrix multiplication, which was inspired by the tiling algorithm that is used to compute Matrix Multiplication on GPU. To utilize on-chip shared memory efficiently, matrices are partitioned into tiles, and each tile does the computation independently. The sparse matrix multiplication can be computed without noticeable overhead if the sparsity is assigned tile-wise manner. We implemented the CUDA-level kernel for tile-spare matrix multiplication by modifying [CUTLASS, NVIDIA](https://github.com/NVIDIA/cutlass).

# Tile-sparse matrix encoding
![ALT](/images/tile_sparse_encoding.png "Tile-sparse matrix encoding")
The tile-sparse matrix is encoded with CSC (Compressed Column Storage)-like format. The difference between the conventional CSC and the tile-sparse CSC is that the basic encoding unit is the tile, not the single data point in the tile-sparse CSC. As shown in the above figure, the “ptr” stores accumulated number of non-zero tiles for each column, the “indices” stores row indices of the tiles, and the “data” stores data on non-zero tiles.


# Performance
![ALT](/images/performance.png "Tile-sparse performance comparison with Block-sparse on Matrix Multiplication. The weight matrix size is 4096x4096, and the minibatch is size of 32. The size of block/tile is 32x32.")
Recently, OpenAI released [Block-sparse GPU Kernels](https://github.com/openai/blocksparse) [2]. Similar to the proposed Tile-sparse, it computes block-wise sparse matrix. We compared the performance of the Block-sparse and the Tile-sparse kernels with TitanXp and CUDA9.0. As the above figure shows, the speed of the Tile-sparse is comparable with the Block-sparse scheme, when applied to a 4096x4096 weight matrix, minibatch size of 32 and block/tile size of 32x32. (Note that the relative speed can be changed according to the matrix size, minibatch size, and the tile size.)

Compared to the Block-sparse GPU which was written in the assembly language level, the Tile-sparse kernel was written in CUDA C++. We believe that such a feature can help researchers to update the kernel further for various flavors. Please contact me(yulhwa.kim@postech.ac.kr) for further information if necessary. Enjoy!


# Makefile & Program Usage
It is as same as [CUTLASS, NVIDIA](https://github.com/NVIDIA/cutlass), but options for the makefile are reduced as follow.

    make <sgemm|dgemm> sm=<60|61> \
      [transpose=<nn>] [verbose=<0|1>] [keep=<0|1>]
      

# Reference
[1] NVIDIA. 2017. CUTLASS. Available at: https://github.com/NVIDIA/cutlass.

[2] OpenAI. 2017. Blocksparse. Available at: https://github.com/openai/blocksparse.
