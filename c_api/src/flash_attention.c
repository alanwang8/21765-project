/**
 * @file flash_attention.c
 * Online softmax attention (Dao et al., 2022) — CPU reference implementation.
 *
 * For each query position we stream over keys and update running max m,
 * normalizer l, and output using the stable online softmax recurrence.
 */

#include "flash_attention.h"

#include <float.h>
#include <math.h>
#include <stddef.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

static inline size_t idx4(int B, int H, int N, int D, int b, int h, int i, int d) {
    (void)B;
    (void)H;
    (void)N;
    return (size_t)(((b * H + h) * N + i) * D) + (size_t)d;
}

int flash_attention_cpu_forward(const float *Q, const float *K, const float *V,
                                float *O, const FlashAttentionParams *p) {
    if (!Q || !K || !V || !O || !p) {
        return FLASH_ATTENTION_ERR_NULL;
    }
    const int B = p->batch;
    const int H = p->heads;
    const int N = p->seq_len;
    const int D = p->head_dim;
    if (B <= 0 || H <= 0 || N <= 0 || D <= 0) {
        return FLASH_ATTENTION_ERR_BAD_DIM;
    }
    if (D > FLASH_ATTENTION_MAX_HEAD_DIM) {
        return FLASH_ATTENTION_ERR_HEAD_DIM;
    }

    const float inv_sqrt_d = 1.0f / sqrtf((float)D);

#ifdef USE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int i = 0; i < N; ++i) {
                float m_i = -FLT_MAX;
                float l_i = 0.0f;
                float o_accum[FLASH_ATTENTION_MAX_HEAD_DIM];

                for (int d = 0; d < D; ++d) {
                    o_accum[d] = 0.0f;
                }

                for (int j = 0; j < N; ++j) {
                    if (p->causal && j > i) {
                        continue;
                    }
                    float s = 0.0f;
                    for (int d = 0; d < D; ++d) {
                        s += Q[idx4(B, H, N, D, b, h, i, d)] * K[idx4(B, H, N, D, b, h, j, d)];
                    }
                    s *= inv_sqrt_d;

                    const float m_prev = m_i;
                    m_i = m_prev > s ? m_prev : s;
                    const float exp_scale_prev = expf(m_prev - m_i);
                    const float exp_s = expf(s - m_i);
                    l_i = l_i * exp_scale_prev + exp_s;

                    for (int d = 0; d < D; ++d) {
                        o_accum[d] =
                            o_accum[d] * exp_scale_prev + exp_s * V[idx4(B, H, N, D, b, h, j, d)];
                    }
                }

                if (l_i <= 0.0f) {
                    l_i = 1.0f;
                }
                for (int d = 0; d < D; ++d) {
                    O[idx4(B, H, N, D, b, h, i, d)] = o_accum[d] / l_i;
                }
            }
        }
    }

    return FLASH_ATTENTION_OK;
}
