// Transpose a nonaliasing row-major float matrix (rows,cols) using a 32x33 shared tile
// and 32x8 threads. Handle partial tiles; dimensions are positive.

#include <cuda_runtime.h>
#include <math.h>

__global__ void transpose(const float *x, float *y, int rows, int cols) {
    __shared__ float tile[32][33];
    int tx = threadIdx.x, ty = threadIdx.y;
    int c = blockIdx.x * 32 + tx, r = blockIdx.y * 32 + ty;
    for (int j = 0; j < 32; j += 8)
        if (c < cols && r + j < rows)
            tile[ty + j][tx] = x[(r + j) * cols + c];
    __syncthreads();
    c = blockIdx.y * 32 + tx;
    r = blockIdx.x * 32 + ty;
    for (int j = 0; j < 32; j += 8)
        if (c < rows && r + j < cols)
            y[(r + j) * rows + c] = tile[tx][ty + j];
}
void solve(const float *x, float *y, int rows, int cols, cudaStream_t s = 0) {
    transpose<<<dim3((cols + 31) / 32, (rows + 31) / 32), dim3(32, 8), 0, s>>>(
        x, y, rows, cols);
}
