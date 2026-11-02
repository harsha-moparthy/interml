// Implement numerically stable row-wise softmax for a finite row-major float matrix.
// Use one 256-thread block per row and shared-memory reductions; row width may exceed
// 256.

#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax(const float *x, float *y, int cols) {
    __shared__ float buf[256];
    int t = threadIdx.x, row = blockIdx.x;
    float mx = -INFINITY;
    for (int j = t; j < cols; j += 256)
        mx = fmaxf(mx, x[row * cols + j]);
    buf[t] = mx;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (t < d)
            buf[t] = fmaxf(buf[t], buf[t + d]);
        __syncthreads();
    }
    mx = buf[0];
    __syncthreads();
    float sum = 0;
    for (int j = t; j < cols; j += 256)
        sum += expf(x[row * cols + j] - mx);
    buf[t] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (t < d)
            buf[t] += buf[t + d];
        __syncthreads();
    }
    for (int j = t; j < cols; j += 256)
        y[row * cols + j] = expf(x[row * cols + j] - mx) / buf[0];
}
void solve(const float *x, float *y, int rows, int cols, cudaStream_t s = 0) {
    if (rows > 0 && cols > 0)
        softmax<<<rows, 256, 0, s>>>(x, y, cols);
}
