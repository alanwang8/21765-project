# Security considerations

## Attack surface

- The libraries perform **numerical computation** on in-memory tensors. There is **no network server**, credential handling, or deserialization of untrusted file formats in the core `c_api` or ndarray backends.
- **DoS via allocation:** Extremely large `batch`, `heads`, `seq_len`, or `head_dim` can cause large `malloc` / GPU allocations or long runtimes. Callers should validate dimensions against application limits. The C API rejects `head_dim > FLASH_ATTENTION_MAX_HEAD_DIM` (512).
- **Numerical stability:** Very large logits can overflow `expf`; inputs should stay in a reasonable float32 range for stable softmax.

## Reporting

For security-sensitive bugs (e.g. unexpected code execution via build scripts), open a GitHub issue or contact the maintainers directly via the repository.
