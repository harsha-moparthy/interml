// Reduce every consecutive group of 32 float elements to one sum using warp shuffle
// instructions. Pad the final partial group with zeros and write ceil(n/32) outputs.

#include <cuda_runtime.h>
#include <math.h>

__global__ void warp_sums(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = i < n ? x[i] : 0;
    for (int offset = 16; offset > 0; offset /= 2)
        v += __shfl_down_sync(0xffffffff, v, offset);
    if (threadIdx.x % 32 == 0 && i < n)
        y[i / 32] = v;
}
void solve(const float *x, float *y, int n, cudaStream_t s = 0) {
    if (n > 0)
        warp_sums<<<(n + 255) / 256, 256, 0, s>>>(x, y, n);
}
