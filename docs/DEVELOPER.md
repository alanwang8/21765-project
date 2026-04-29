# Developer guide

## Algorithm

**Flash-style attention** uses an **online softmax** over key positions so attention weights need not materialize an full \(N \times N\) matrix in memory (Dao et al., 2022). This repository contains:

1. **CUDA** — `FlashAttentionForwardKernel` / `FlashAttentionBackwardKernel` in [src/ndarray_backend_cuda.cu](../src/ndarray_backend_cuda.cu): tiled loops over sequence blocks with running statistics \(m\), \(l\) and masked causal attention.
2. **Python** — [python/needle/ops/ops_attention.py](../python/needle/ops/ops_attention.py) wraps device calls and caches \(m\), \(l\) on the forward output for backward.
3. **CPU C reference** — [c_api/src/flash_attention.c](../c_api/src/flash_attention.c): per-\((batch, head)\) online softmax over all keys (correctness reference; not IO-aware like the paper’s GPU kernel).

## Module map

| Component | Role |
|-----------|------|
| `ops_attention.py` | `FlashAttention`, `FlashAttentionGrad`, `flash_attention()` |
| `nn_transformer.py` | `MultiHeadAttention` optional `use_flash_attention` |
| `ndarray_backend_cuda.cu` | pybind exports `flash_attention_forward/backward` |
| `c_api/*` | Portable CPU forward for tests and pedagogy |

## Extending

- **Fused CUDA:** replace inner loops with shared-memory tiling and warp primitives; keep Python tensor shapes unchanged.
- **C API:** add backward in a new `.c` file and expose through `flash_attention.h` with documented buffer ownership.
- **OpenMP:** compile with `-DUSE_OPENMP -fopenmp` (see [INSTALL.md](INSTALL.md)); outer `parallel for` spans `batch × heads`.

## Coding style

- C11 for `c_api/`; run `make format` for Python and clang-format on `src/*.cc` / `src/*.cu`.
