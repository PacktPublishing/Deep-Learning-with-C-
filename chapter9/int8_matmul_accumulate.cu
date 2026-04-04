// int8_matmul_accumulate.cu
// Demonstrates how modern GPU integer MAC (multiply-accumulate) units work:
//   - Multiply 8-bit integer operands from quantized weight/activation matrices
//   - Accumulate products in 32-bit registers to prevent overflow
//   - Dequantize the 32-bit result back to floating point
//
// Compile: nvcc -O2 -std=c++17 int8_matmul_accumulate.cu -o int8_matmul_accumulate
// Run:     ./int8_matmul_accumulate

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Quantization helpers (host side)
// ---------------------------------------------------------------------------

struct QuantParams {
    float scale;
    int32_t zero_point;
};

// Asymmetric affine quantization: q = clamp(round(x / scale) + zp, 0, 255)
QuantParams compute_quant_params(const float* data, int n) {
    float mn = data[0], mx = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    float scale = (mx - mn) / 255.0f;
    if (scale == 0.0f) scale = 1.0f;
    int32_t zp = (int32_t)roundf(-mn / scale);
    if (zp < 0) zp = 0;
    if (zp > 255) zp = 255;
    return {scale, zp};
}

void quantize_to_int8(const float* src, uint8_t* dst, int n, QuantParams p) {
    for (int i = 0; i < n; ++i) {
        int32_t q = (int32_t)roundf(src[i] / p.scale) + p.zero_point;
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        dst[i] = (uint8_t)q;
    }
}

// ---------------------------------------------------------------------------
// CUDA kernel: INT8 matmul with INT32 accumulation
//
// C_int32[M,N] = A_uint8[M,K] * B_uint8[K,N]   (dot-product per element)
//
// Each thread computes one element of C. The multiply of two uint8 values
// fits in 16 bits, but we accumulate K such products in a 32-bit register
// so the running sum never overflows (K can be up to ~16 million before
// overflow with worst-case uint8*uint8 = 65025 per product).
// ---------------------------------------------------------------------------

__global__ void int8_matmul_int32_accum(const uint8_t* __restrict__ A,
                                         const uint8_t* __restrict__ B,
                                         int32_t* __restrict__ C,
                                         int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    // 32-bit accumulator — the "larger bucket" that holds the growing sum
    int32_t acc = 0;
    for (int k = 0; k < K; ++k) {
        // Each multiply: uint8 * uint8 -> up to 16 bits
        // Accumulated in 32-bit register -> safe for K up to millions
        acc += (int32_t)A[row * K + k] * (int32_t)B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Tiled shared-memory version for better performance
// Tiles of A and B are loaded into shared memory in uint8, accumulated in int32.
// ---------------------------------------------------------------------------

#define TILE 16

__global__ void int8_matmul_tiled(const uint8_t* __restrict__ A,
                                   const uint8_t* __restrict__ B,
                                   int32_t* __restrict__ C,
                                   int M, int N, int K) {
    __shared__ uint8_t As[TILE][TILE];
    __shared__ uint8_t Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    int32_t acc = 0;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += (int32_t)As[threadIdx.y][k] * (int32_t)Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// Dequantization kernel: convert INT32 accumulator back to FP32
//
// real_value ≈ scale_A * scale_B * (acc - correction_terms)
// For simplicity we apply the full affine correction:
//   C_float[i,j] = sA * sB * (C_int32[i,j]
//                  - zpB * rowsum_A[i] - zpA * colsum_B[j]
//                  + K * zpA * zpB)
// ---------------------------------------------------------------------------

__global__ void dequantize_int32_to_fp32(const int32_t* __restrict__ C_int,
                                          float* __restrict__ C_fp,
                                          const int32_t* __restrict__ row_sum_A,
                                          const int32_t* __restrict__ col_sum_B,
                                          float sA, float sB,
                                          int32_t zpA, int32_t zpB,
                                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int32_t raw = C_int[row * N + col];
    // Affine correction to undo zero-point bias
    int32_t corrected = raw - zpB * row_sum_A[row] - zpA * col_sum_B[col]
                        + K * zpA * zpB;
    C_fp[row * N + col] = sA * sB * (float)corrected;
}

// ---------------------------------------------------------------------------
// Host helpers: compute row sums / col sums of uint8 matrices
// ---------------------------------------------------------------------------

void row_sums(const uint8_t* mat, int32_t* sums, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        int32_t s = 0;
        for (int c = 0; c < cols; ++c) s += mat[r * cols + c];
        sums[r] = s;
    }
}

void col_sums(const uint8_t* mat, int32_t* sums, int rows, int cols) {
    for (int c = 0; c < cols; ++c) {
        int32_t s = 0;
        for (int r = 0; r < rows; ++r) s += mat[r * cols + c];
        sums[c] = s;
    }
}

// FP32 reference matmul on host
void matmul_fp32_ref(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

// ---------------------------------------------------------------------------
// Main: end-to-end INT8 matmul pipeline
// ---------------------------------------------------------------------------

int main() {
    const int M = 128, K = 256, N = 64;
    printf("INT8 Matrix Multiply with 32-bit Accumulation\n");
    printf("==============================================\n");
    printf("Matrix dimensions: A[%d x %d] * B[%d x %d] = C[%d x %d]\n\n",
           M, K, K, N, M, N);

    // --- Allocate and fill FP32 matrices with small random values ---
    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC_ref = new float[M * N];
    srand(42);
    for (int i = 0; i < M * K; ++i) hA[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < K * N; ++i) hB[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    // FP32 reference
    matmul_fp32_ref(hA, hB, hC_ref, M, N, K);

    // --- Quantize A and B to uint8 ---
    QuantParams qA = compute_quant_params(hA, M * K);
    QuantParams qB = compute_quant_params(hB, K * N);
    printf("Quantization parameters:\n");
    printf("  A: scale=%.6f  zero_point=%d\n", qA.scale, qA.zero_point);
    printf("  B: scale=%.6f  zero_point=%d\n\n", qB.scale, qB.zero_point);

    uint8_t *hA_q = new uint8_t[M * K];
    uint8_t *hB_q = new uint8_t[K * N];
    quantize_to_int8(hA, hA_q, M * K, qA);
    quantize_to_int8(hB, hB_q, K * N, qB);

    // Row sums of A_q, col sums of B_q (for dequantization correction)
    int32_t *hRowSumA = new int32_t[M];
    int32_t *hColSumB = new int32_t[N];
    row_sums(hA_q, hRowSumA, M, K);
    col_sums(hB_q, hColSumB, K, N);

    // --- GPU allocations ---
    uint8_t *dA, *dB;
    int32_t *dC_int, *dRowSumA, *dColSumB;
    float *dC_fp;
    CHECK_CUDA(cudaMalloc(&dA, M * K));
    CHECK_CUDA(cudaMalloc(&dB, K * N));
    CHECK_CUDA(cudaMalloc(&dC_int, M * N * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dC_fp, M * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dRowSumA, M * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&dColSumB, N * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(dA, hA_q, M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB_q, K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dRowSumA, hRowSumA, M * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dColSumB, hColSumB, N * sizeof(int32_t), cudaMemcpyHostToDevice));

    // --- Launch naive INT8 matmul kernel ---
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm up
    int8_matmul_int32_accum<<<grid, block>>>(dA, dB, dC_int, M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark naive kernel
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; ++i)
        int8_matmul_int32_accum<<<grid, block>>>(dA, dB, dC_int, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_naive = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_naive, start, stop));
    printf("Naive INT8 kernel:  %.3f ms (avg over 100 runs)\n", ms_naive / 100.0f);

    // Benchmark tiled kernel
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < 100; ++i)
        int8_matmul_tiled<<<grid, block>>>(dA, dB, dC_int, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms_tiled = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tiled, start, stop));
    printf("Tiled INT8 kernel:  %.3f ms (avg over 100 runs)\n", ms_tiled / 100.0f);
    printf("Speedup (tiled/naive): %.2fx\n\n", ms_naive / ms_tiled);

    // Use tiled result for accuracy check
    int8_matmul_tiled<<<grid, block>>>(dA, dB, dC_int, M, N, K);

    // --- Dequantize INT32 accumulator -> FP32 ---
    dequantize_int32_to_fp32<<<grid, block>>>(
        dC_int, dC_fp, dRowSumA, dColSumB,
        qA.scale, qB.scale, qA.zero_point, qB.zero_point,
        M, N, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    float *hC_int8 = new float[M * N];
    CHECK_CUDA(cudaMemcpy(hC_int8, dC_fp, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Accuracy comparison vs FP32 reference ---
    double max_err = 0, sum_err = 0, sum_ref = 0;
    for (int i = 0; i < M * N; ++i) {
        double err = fabs((double)hC_int8[i] - (double)hC_ref[i]);
        if (err > max_err) max_err = err;
        sum_err += err;
        sum_ref += fabs((double)hC_ref[i]);
    }
    double mean_err = sum_err / (M * N);
    double rel_err = sum_err / (sum_ref + 1e-12);

    printf("Accuracy (INT8 vs FP32 reference):\n");
    printf("  Max absolute error:  %.6f\n", max_err);
    printf("  Mean absolute error: %.6f\n", mean_err);
    printf("  Relative error:      %.4f%%\n\n", rel_err * 100.0);

    // --- Memory savings ---
    long fp32_bytes = (long)(M * K + K * N) * 4;
    long int8_bytes = (long)(M * K + K * N) * 1;
    printf("Memory footprint (A + B):\n");
    printf("  FP32: %ld bytes\n", fp32_bytes);
    printf("  INT8: %ld bytes (%.1fx reduction)\n\n", int8_bytes,
           (float)fp32_bytes / int8_bytes);

    // --- Show overflow demonstration ---
    printf("Why 32-bit accumulation is necessary:\n");
    printf("  Max product of two uint8 values: 255 * 255 = %d (needs 16 bits)\n", 255 * 255);
    printf("  Accumulating K=%d such products: max sum = %lld\n", K,
           (long long)K * 255 * 255);
    printf("  16-bit max: %d  -> would overflow after %d additions\n",
           65535, 65535 / (255 * 255));
    printf("  32-bit max: %u -> safe for K up to %lld\n",
           UINT32_MAX, (long long)UINT32_MAX / (255 * 255));

    // Print a small tile of results for visual inspection
    printf("\nSample output (top-left 4x4):\n");
    printf("  %-12s %-12s\n", "FP32 ref", "INT8 result");
    for (int i = 0; i < 4 && i < M; ++i) {
        for (int j = 0; j < 4 && j < N; ++j)
            printf("  %8.3f/%8.3f", hC_ref[i * N + j], hC_int8[i * N + j]);
        printf("\n");
    }

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_int));
    CHECK_CUDA(cudaFree(dC_fp));
    CHECK_CUDA(cudaFree(dRowSumA));
    CHECK_CUDA(cudaFree(dColSumB));
    delete[] hA; delete[] hB; delete[] hC_ref; delete[] hC_int8;
    delete[] hA_q; delete[] hB_q; delete[] hRowSumA; delete[] hColSumB;

    printf("\nDone.\n");
    return 0;
}
