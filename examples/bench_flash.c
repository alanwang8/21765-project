/**
 * @file bench_flash.c
 * Small driver: run CPU flash attention on a few shapes and print a checksum.
 * Demonstrates non-trivial sequence lengths and multi-head layout.
 */

#include "flash_attention.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static float checksum(const float *x, size_t n) {
    float s = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        s += x[i] * (float)(i % 31 + 1);
    }
    return s;
}

int main(void) {
    const int B = 1, H = 4, D = 64;
    const int lengths[] = {128, 512, 1024};
    const int nlen = (int)(sizeof(lengths) / sizeof(lengths[0]));

    for (int li = 0; li < nlen; ++li) {
        int N = lengths[li];
        size_t total = (size_t)B * H * N * D;
        float *Q = (float *)malloc(total * sizeof(float));
        float *K = (float *)malloc(total * sizeof(float));
        float *V = (float *)malloc(total * sizeof(float));
        float *O = (float *)malloc(total * sizeof(float));
        if (!Q || !K || !V || !O) {
            fprintf(stderr, "allocation failed\n");
            return 1;
        }
        for (size_t i = 0; i < total; ++i) {
            Q[i] = (float)(i % 997) * 1e-3f;
            K[i] = (float)(i % 991) * 1e-3f;
            V[i] = (float)(i % 983) * 1e-3f;
        }

        FlashAttentionParams p = {
            .batch = B, .heads = H, .seq_len = N, .head_dim = D, .block_m = 128, .block_n = 128, .causal = 1};

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int st = flash_attention_cpu_forward(Q, K, V, O, &p);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (st != FLASH_ATTENTION_OK) {
            fprintf(stderr, "forward failed: %d (N=%d)\n", st, N);
            free(Q);
            free(K);
            free(V);
            free(O);
            return 1;
        }
        double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        printf("N=%5d  causal=1  time=%.4fs  checksum=%.6g\n", N, sec, (double)checksum(O, total));

        free(Q);
        free(K);
        free(V);
        free(O);
    }
    return 0;
}
