// Compute RMSNorm(x)*weight for each row of a float matrix using one block per row and
// positive epsilon. All pointers refer to contiguous device arrays.

#include <cuda_runtime.h>
#include <math.h>

__global__ void rmsnorm(const float *x, const float *weight, float *y, int cols,
                        float eps) {
    __shared__ float buf[256];
    int t = threadIdx.x, row = blockIdx.x;
    float sum = 0;
    for (int j = t; j < cols; j += 256) {
        float v = x[row * cols + j];
        sum += v * v;
    }
    buf[t] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (t < d)
            buf[t] += buf[t + d];
        __syncthreads();
    }
    float inv = rsqrtf(buf[0] / cols + eps);
    for (int j = t; j < cols; j += 256)
        y[row * cols + j] = x[row * cols + j] * inv * weight[j];
}
void solve(const float *x, const float *w, float *y, int rows, int cols,
           float eps = 1e-5f, cudaStream_t s = 0) {
    if (rows > 0 && cols > 0)
        rmsnorm<<<rows, 256, 0, s>>>(x, w, y, cols, eps);
}
