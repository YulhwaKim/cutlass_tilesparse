# cutlass_tilesparse
Cuda templates for tile-sparse matrix multiplication based on [CUTLASS](https://github.com/NVIDIA/cutlass).


# Introduction
Since Matrix Multiplication occupy most of the Neural Network computations, diversification and optimization on Matrix Multiplication kernels can give inspiration for the efficient Neural Network design.

In this project, we developed Tile-sparse Matrix multiplication. It is inspired by the tiling algorithm that is used to compute Matrix Multiplication on GPU. To utilize on-chip shared memory efficiently, matrices are partitioned into tiles, and each tile do the computation independently. For this reason, if we assign sparsity in tile-wise manner, then we can compute sparse matrix multiplication without noticeable overhead. We implemented the CUDA-level kernel for tile-spare matrix multiplication by hacking [CUTLASS](https://github.com/NVIDIA/cutlass).


# Tile-sparse matrix encoding
![ALT](/images/bsc.png | width 100)
The tile-sparse matrix is encoded with CSC (Compressed Column Storage). The difference between the conventional CSC and tile-sparse CSC is that the basic encoding unit is the tile, not the single data point.
As the above figure, ptr stores accumulated non-zero tiles for each column, indices stores row indices of each tile, and data stores data on non-zero tiles.


# Performance
![ALT](/images/performance.png "Tile-sparse performance comparison with Block-sparse on Matrix Multiplication. The weight matrix size is 4096x4096, and the minibatch is size of 32. The size of block/tile is 32x32.")
Recently, OpenAI released [Block-sparse GPU Kernels](https://github.com/openai/blocksparse). As Tile-sparse, it computes block-wise sparse matrix. We compared the performance of Block-sparse and Tile-sparse kernels with TitanXp and CUDA9.0. As the above figure shows, Tile-sparse is slightly faster than Block-sparse, when used a 4096x4096 weight matrix, minibatch size of 32 and block/tile size of 32x32. (The relative speed can be changed according to the matrix size, minibatch size, and the tile size.)


# Makefile & Program Usage
It is as same as [CUTLASS](https://github.com/NVIDIA/cutlass), but options for the makefile are reduced as follow.

    make <sgemm|dgemm> sm=<60|61> \
      [transpose=<nn>] [verbose=<0|1>] [keep=<0|1>]
