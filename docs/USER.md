# User guide — Flash attention in Needle

## Python API

After building backends (`make lib`), add the repository’s `python/` directory to `PYTHONPATH` (or install the package in editable mode if you add a `setup.cfg`).

```python
import needle as ndl
from needle import ops

# Q, K, V: ndl.Tensor on CUDA, shape (batch, heads, seq_len, head_dim)
out = ops.flash_attention(q, k, v, causal=False, block_m=128, block_n=128)
```

### Requirements

- **Device:** Flash attention in this tree is implemented on **CUDA** (`ops_attention.FlashAttention` calls into the CUDA backend). Use `MultiHeadAttention(..., use_flash_attention=True)` in `needle.nn.nn_transformer` for the transformer path.

## CPU C API (reference)

For a standalone numerical core without Python, include [c_api/include/flash_attention.h](../c_api/include/flash_attention.h) and link the object produced by `make c-api`.

```c
FlashAttentionParams p = { .batch = 1, .heads = 1, .seq_len = 64,
  .head_dim = 32, .block_m = 128, .block_n = 128, .causal = 0 };
flash_attention_cpu_forward(Q, K, V, O, &p);
```

Layouts are **row-major** compact `float` tensors with index

`(((batch * heads + head) * seq_len + pos) * head_dim) + d`.

Maximum `head_dim` is `FLASH_ATTENTION_MAX_HEAD_DIM` (512). Return codes are listed in the header.

## Examples

- `examples/bench_flash.c` — built by `make example-c`.
- Technical narrative and benchmarks: see `Flash_Attention_Final_Report.ipynb` in the repository root.
