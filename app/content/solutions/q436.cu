// Compute an integer histogram with atomic adds. Values outside [0,bins) are ignored.
// Initialize the device output to zero on the supplied stream; bins must be positive.

#include <cuda_runtime.h>
#include <math.h>

__global__ void histogram(const int *x, int *out, int n, int bins) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        int v = x[i];
        if (v >= 0 && v < bins)
            atomicAdd(out + v, 1);
    }
}
void solve(const int *x, int *out, int n, int bins, cudaStream_t s = 0) {
    cudaMemsetAsync(out, 0, bins * sizeof(int), s);
    if (n > 0)
        histogram<<<(n + 255) / 256, 256, 0, s>>>(x, out, n, bins);
}
