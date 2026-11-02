// Implement an educational tiled online-softmax attention kernel for one causal head
// Q,K,V(N,D), with 1<=D<=128. Each thread handles one query and streams key tiles,
// avoiding an N-by-N score matrix. This emphasizes the FlashAttention recurrence, not
// production throughput.

#include <cuda_runtime.h>
#include <math.h>

__global__ void attention(const float *q, const float *k, const float *v, float *out,
                          int n, int d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n)
        return;
    float acc[128];
    for (int c = 0; c < d; ++c)
        acc[c] = 0;
    float maximum = -INFINITY, denom = 0;
    for (int tile = 0; tile <= row; tile += 32) {
        for (int j = tile; j < tile + 32 && j <= row; ++j) {
            float score = 0;
            for (int c = 0; c < d; ++c)
                score += q[row * d + c] * k[j * d + c];
            score /= sqrtf((float)d);
            float next = fmaxf(maximum, score);
            float old = expf(maximum - next), p = expf(score - next);
            for (int c = 0; c < d; ++c)
                acc[c] = old * acc[c] + p * v[j * d + c];
            denom = old * denom + p;
            maximum = next;
        }
    }
    for (int c = 0; c < d; ++c)
        out[row * d + c] = acc[c] / denom;
}
void solve(const float *q, const float *k, const float *v, float *out, int n, int d,
           cudaStream_t s = 0) {
    if (n > 0 && d > 0 && d <= 128)
        attention<<<(n + 63) / 64, 64, 0, s>>>(q, k, v, out, n, d);
}
