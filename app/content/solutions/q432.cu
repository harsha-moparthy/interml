// Square n float elements on device using a grid-stride loop, capping the grid at 128
// blocks. Support n=0; x and y may alias.

#include <cuda_runtime.h>
#include <math.h>

__global__ void square(const float *x, float *y, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        y[i] = x[i] * x[i];
}
void solve(const float *x, float *y, int n, cudaStream_t s = 0) {
    int blocks = (n + 255) / 256;
    if (blocks > 128)
        blocks = 128;
    if (n > 0)
        square<<<blocks, 256, 0, s>>>(x, y, n);
}
