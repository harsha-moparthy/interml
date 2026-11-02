// Write a CUDA kernel and launcher that fill a device int array with global thread
// indices. Accept n>=0 and use 256 threads per block; launch asynchronously on the
// supplied stream.

#include <cuda_runtime.h>
#include <math.h>

__global__ void indices(int *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = i;
}
void solve(int *out, int n, cudaStream_t stream = 0) {
    if (n > 0)
        indices<<<(n + 255) / 256, 256, 0, stream>>>(out, n);
}
