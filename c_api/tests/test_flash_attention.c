/**
 * @file test_flash_attention.c
 * Compare CPU Flash-style forward to naive softmax attention on small tensors.
 */

#include "flash_attention.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t idx4(int B, int H, int N, int D, int b, int h, int i, int j) {
    (void)B;
    (void)H;
    (void)N;
    return (size_t)(((b * H + h) * N + i) * D) + (size_t)j;
}

static void naive_attention(const float *Q, const float *K, const float *V, float *O,
                            int B, int H, int N, int D, int causal) {
    const float scale = 1.0f / sqrtf((float)D);
    float *scores = (float *)malloc((size_t)N * sizeof(float));
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < N; ++i) {
                float m = -FLT_MAX;
                float sum = 0.0f;
                for (int j = 0; j < N; ++j) {
                    if (causal && j > i) {
                        scores[j] = -INFINITY;
                    } else {
                        float s = 0.0f;
                        for (int d = 0; d < D; ++d) {
                            s += Q[idx4(B, H, N, D, b, h, i, d)] * K[idx4(B, H, N, D, b, h, j, d)];
                        }
                        s *= scale;
                        scores[j] = s;
                        if (s > m) {
                            m = s;
                        }
                    }
                }
                for (int j = 0; j < N; ++j) {
                    float w = expf(scores[j] - m);
                    scores[j] = w;
                    sum += w;
                }
                if (sum <= 0.0f) {
                    sum = 1.0f;
                }
                for (int d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j < N; ++j) {
                        acc += scores[j] / sum * V[idx4(B, H, N, D, b, h, j, d)];
                    }
                    O[idx4(B, H, N, D, b, h, i, d)] = acc;
                }
            }
        }
    }
    free(scores);
}

static int nearly_equal(float a, float b, float atol, float rtol) {
    float diff = fabsf(a - b);
    return diff <= atol + rtol * fabsf(b);
}

int main(void) {
    const int B = 2, H = 2, N = 17, D = 32;
    const size_t total = (size_t)B * H * N * D;
    float *Q = (float *)malloc(total * sizeof(float));
    float *K = (float *)malloc(total * sizeof(float));
    float *V = (float *)malloc(total * sizeof(float));
    float *O_flash = (float *)malloc(total * sizeof(float));
    float *O_naive = (float *)malloc(total * sizeof(float));
    if (!Q || !K || !V || !O_flash || !O_naive) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    for (size_t i = 0; i < total; ++i) {
        Q[i] = (float)(i % 13) * 0.01f - 0.05f;
        K[i] = (float)(i % 7) * 0.02f;
        V[i] = (float)(i % 11) * 0.03f - 0.1f;
    }

    FlashAttentionParams p = {.batch = B,
                              .heads = H,
                              .seq_len = N,
                              .head_dim = D,
                              .block_m = 128,
                              .block_n = 128,
                              .causal = 0};
    memset(O_flash, 0, total * sizeof(float));
    memset(O_naive, 0, total * sizeof(float));

    int st = flash_attention_cpu_forward(Q, K, V, O_flash, &p);
    if (st != FLASH_ATTENTION_OK) {
        fprintf(stderr, "flash_attention_cpu_forward failed: %d\n", st);
        return 1;
    }
    naive_attention(Q, K, V, O_naive, B, H, N, D, 0);

    const float atol = 1e-4f;
    const float rtol = 1e-3f;
    for (size_t i = 0; i < total; ++i) {
        if (!nearly_equal(O_flash[i], O_naive[i], atol, rtol)) {
            fprintf(stderr, "Mismatch at %zu: flash=%g naive=%g\n", i, (double)O_flash[i],
                    (double)O_naive[i]);
            free(Q);
            free(K);
            free(V);
            free(O_flash);
            free(O_naive);
            return 1;
        }
    }

    /* Causal smoke test */
    p.causal = 1;
    st = flash_attention_cpu_forward(Q, K, V, O_flash, &p);
    if (st != FLASH_ATTENTION_OK) {
        fprintf(stderr, "causal forward failed: %d\n", st);
        return 1;
    }
    naive_attention(Q, K, V, O_naive, B, H, N, D, 1);
    for (size_t i = 0; i < total; ++i) {
        if (!nearly_equal(O_flash[i], O_naive[i], atol, rtol)) {
            fprintf(stderr, "Causal mismatch at %zu: flash=%g naive=%g\n", i, (double)O_flash[i],
                    (double)O_naive[i]);
            free(Q);
            free(K);
            free(V);
            free(O_flash);
            free(O_naive);
            return 1;
        }
    }

    free(Q);
    free(K);
    free(V);
    free(O_flash);
    free(O_naive);
    printf("test_flash_attention: PASS\n");
    return 0;
}
