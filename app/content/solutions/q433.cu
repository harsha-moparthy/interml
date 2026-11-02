// Multiply nonaliasing row-major float matrices A(M,K) and B(K,N) into C(M,N).
// Implement 16x16 shared-memory tiles with masked boundary loads; dimensions are
// positive.

#include <cuda_runtime.h>
#include <math.h>

__global__ void matmul(const float *a, const float *b, float *c, int m, int n, int k) {
    __shared__ float as[16][16], bs[16][16];
    int x = threadIdx.x, y = threadIdx.y;
    int row = blockIdx.y * 16 + y, col = blockIdx.x * 16 + x;
    float sum = 0;
    for (int t = 0; t < (k + 15) / 16; ++t) {
        as[y][x] = (row < m && t * 16 + x < k) ? a[row * k + t * 16 + x] : 0;
        bs[y][x] = (t * 16 + y < k && col < n) ? b[(t * 16 + y) * n + col] : 0;
        __syncthreads();
        for (int j = 0; j < 16; ++j)
            sum += as[y][j] * bs[j][x];
        __syncthreads();
    }
    if (row < m && col < n)
        c[row * n + col] = sum;
}
void solve(const float *a, const float *b, float *c, int m, int n, int k,
           cudaStream_t s = 0) {
    matmul<<<dim3((n + 15) / 16, (m + 15) / 16), dim3(16, 16), 0, s>>>(a, b, c, m, n,
                                                                       k);
}
