/**
 * @file flash_attention.h
 * @brief CPU reference: scaled dot-product attention with online softmax (Flash-style).
 *
 * Layout for Q, K, V, O is row-major compact float32:
 *   index = (((batch * num_heads + head) * seq_len + pos) * head_dim) + d
 *
 * This library is independent of the Needle Python/CUDA stack and is intended
 * for correctness checks, teaching, and hosts without a GPU.
 */

#ifndef FLASH_ATTENTION_H
#define FLASH_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Status codes returned by flash_attention_cpu_forward. */
enum FlashAttentionStatus {
    FLASH_ATTENTION_OK = 0,
    FLASH_ATTENTION_ERR_NULL = -1,
    FLASH_ATTENTION_ERR_BAD_DIM = -2,
    /** head_dim larger than ::FLASH_ATTENTION_MAX_HEAD_DIM */
    FLASH_ATTENTION_ERR_HEAD_DIM = -3,
};

/** Maximum head dimension supported by the CPU reference (stack buffer). */
#define FLASH_ATTENTION_MAX_HEAD_DIM 512

/** @brief Problem dimensions and options (no pointers). */
typedef struct FlashAttentionParams {
    int batch;     /**< B */
    int heads;     /**< H */
    int seq_len;   /**< N */
    int head_dim;  /**< D */
    int block_m;   /**< Reserved; tiling uses full rows in this CPU reference */
    int block_n;   /**< Reserved */
    int causal;    /**< Non-zero for causal (autoregressive) masking */
} FlashAttentionParams;

/**
 * @brief Forward pass: O = softmax(mask(QK^T / sqrt(D))) V
 *
 * @param Q Queries, shape (B,H,N,D)
 * @param K Keys, same shape
 * @param V Values, same shape
 * @param O Output buffer, same shape as Q (written by callee)
 * @param p Non-null parameter struct; all dimensions must be positive
 * @return ::FLASH_ATTENTION_OK or an error code
 */
int flash_attention_cpu_forward(const float *Q, const float *K, const float *V,
                                float *O, const FlashAttentionParams *p);

#ifdef __cplusplus
}
#endif

#endif /* FLASH_ATTENTION_H */
