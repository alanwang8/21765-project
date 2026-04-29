# Needle Flash Attention

Memory-efficient **scaled dot-product attention** (Flash-style online softmax) integrated into the **Needle** ndarray framework: Python `TensorOp` API, CUDA implementation for training, and a standalone **CPU C reference** under `c_api/`.

**Authors:** Alan Wang · Zhanming (Jerry) Liang

## Webpage: [https://alanwang8.github.io/21765-project/](url)

---

## Quick links

| Audience | Document |
|----------|----------|
| Build and platforms | [docs/INSTALL.md](docs/INSTALL.md) |
| Using Flash attention from Python | [docs/USER.md](docs/USER.md) |
| Algorithms and code map | [docs/DEVELOPER.md](docs/DEVELOPER.md) |
| Project webpage | [docs/index.html](docs/index.html) |
| Man page (library API) | [man/flash_attention.3](man/flash_attention.3) |

---

## Abstract

We implement **Flash Attention** (Dao et al., 2022) in the **Needle** deep learning framework — replacing the standard O(N²) attention matrix with an online softmax recurrence that reduces working memory to O(N). Our contribution includes a tiled CUDA forward and backward pass wired into Needle's autograd engine, a portable C reference implementation for CPU testing, and a drop-in `use_flash_attention` flag on `MultiHeadAttention`. Correctness is verified against pre-computed reference outputs across batch sizes, sequence lengths, head configurations, and causal masking modes.

---

## Background: The Quadratic Memory Problem

Standard attention computes O = softmax(QKᵀ/√d)·V, which requires storing the full N×N score matrix — **O(N²) memory per head**. Flash Attention avoids this entirely by streaming over blocks and maintaining only two running scalars per query row. The output is *bit-identical* to standard attention — it is not an approximation.

| Sequence Length N | Standard (N×N matrix) | Flash (stats only) | Reduction |
|------------------:|----------------------:|-------------------:|----------:|
| 128               | 256 KB                | 4 KB               | 64×       |
| 512               | 4 MB                  | 16 KB              | 256×      |
| 1 024             | 16 MB                 | 32 KB              | 512×      |
| 4 096             | 256 MB                | 128 KB             | 2 048×    |

*Per head (B=1, H=1), float32. Standard: N²×4 B. Flash: N×2×4 B (running max m and normalizer l).*

---

## Algorithm

Instead of computing all scores first then normalizing, Flash Attention maintains a running maximum mᵢ and normalizer lᵢ per query row. When a larger score arrives, the accumulated output is rescaled by exp(m_prev − mᵢ) — keeping results numerically stable without an N×N buffer.

**Forward — Online Softmax Recurrence**

```
for each (batch b, head h):
  for query row i = 0…N−1:
    mᵢ ← −∞,  lᵢ ← 0,  oᵢ ← 0       # init running max, denom, output
    for key col j = 0…N−1:            # (skip j > i if causal)
      sᵢⱼ   ← Q[i] · K[j] / √D
      m_prev ← mᵢ ;  mᵢ ← max(m_prev, sᵢⱼ)
      Pᵢⱼ   ← exp(sᵢⱼ − mᵢ)
      lᵢ    ← exp(m_prev − mᵢ) · lᵢ + Pᵢⱼ
      oᵢ    ← exp(m_prev − mᵢ) · oᵢ + Pᵢⱼ · V[j]
    O[i] ← oᵢ / lᵢ                    # persist mᵢ, lᵢ for backward
```

The **backward pass** reuses saved (m, l) to recompute attention weights without storing the N×N probability matrix. Two CUDA kernels run sequentially: (1) *ComputeDelta* — row-wise δᵢ = Σⱼ Pᵢⱼ · (∂Oᵢ·V[j]); (2) *AccumulateGradients* — ∂Q, ∂K, ∂V via recomputed Pᵢⱼ, using `atomicAdd` for ∂K and ∂V.

---

## Implementation

Four layers from user API to hardware, with the CUDA path for training and a standalone C path for correctness testing:

```
┌─────────────────────────────────────────────────────────────┐
│  Python Neural Network Modules                              │
│  nn.TransformerLayer · nn.AttentionLayer · nn.MultiHead...  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Autograd Operation                                         │
│  ops.flash_attention(Q, K, V, causal=True)                  │
│  — caches m, l on output NDArray for backward               │
└──────────────────────┬─────────────────────────┬────────────┘
                       ↓                         ↓
         ┌─────────────────────┐   ┌─────────────────────────┐
         │  CUDA Backend       │   │  C CPU Reference        │
         │  Forward + Backward │   │  flash_attention_cpu_   │
         │  ndarray_backend_   │   │  forward()              │
         │  cuda.cu            │   │  c_api/ · OpenMP opt.   │
         └─────────────────────┘   └─────────────────────────┘
```

The forward CUDA kernel launches on grid `dim3(batch, heads)` with block width `head_dim` — each thread owns one element of the head dimension and accumulates a register scalar while streaming over all key positions. `FlashAttention` extends Needle's `TensorOp`, attaching (m, l) to the output so the backward pass can recompute attention weights without re-running the forward.

---

## API & Module Map

| Component | Role |
|-----------|------|
| `ops_attention.py` | `FlashAttention`, `FlashAttentionGrad`, `flash_attention()` |
| `nn_transformer.py` | `MultiHeadAttention` with optional `use_flash_attention` |
| `ndarray_backend_cuda.cu` | pybind exports `flash_attention_forward/backward` |
| `c_api/` | Portable CPU forward for tests |

**needle.nn**

| Module | Signature | I/O |
|--------|-----------|-----|
| `TransformerLayer` | `q_features, num_head, dim_head, hidden_size, dropout=0., causal=True` | (B, T, D) → attn + FFN + residual + norm |
| `AttentionLayer` | `q_features, num_head, dim_head, use_flash_attention=False, causal=True` | (B, T, D) → proj + attn + out proj |
| `MultiHeadAttention` | `causal=False, dropout=0., use_flash_attention=False` | (B, H, T, d) × 3 → (B, H, T, d) |

When `use_flash_attention=True`, delegates to:

**needle.ops** → `flash_attention(Q, K, V, causal=False, block_m=128, block_n=None)`
- CUDA only · in/out (B, H, T, d) · attention probabilities not returned (would negate memory savings)

---

## Results

### CPU Reference Benchmark

Single-threaded, causal masking, B=1 H=4 D=64. Run with `make example-c`.

| N     | Time     | N²          | Ratio vs N=128 |
|------:|--------:|------------:|---------------:|
| 128   | 0.0013 s | 16 384      | 1×             |
| 512   | 0.0205 s | 262 144     | 15.8×          |
| 1 024 | 0.0554 s | 1 048 576   | 42.6×          |

Scales O(N²·D) as expected (theoretical 16× and 64×; observed slightly lower due to cache effects at small N).

### Correctness Verification

The Python test suite validates `MultiHeadAttention`, `AttentionLayer`, `TransformerLayer`, and `Transformer` against pre-computed float32 reference outputs (`atol=1e-5`), parametrized over:
- batch ∈ {4, 8}, seq_len ∈ {5, 11, 31}, heads ∈ {4, 5, 8}, head_dim ∈ {8, 32, 64}
- causal ∈ {True, False}, device ∈ {cpu, cuda}

The C reference test (`make test-c`) additionally verifies online softmax vs. naive attention at `atol=1e-4`.

---

## Quick Start

```bash
# Build Needle backends (requires CMake, pybind11)
make lib

# C reference — no Python/CUDA required
make c-api && make test-c && make example-c
```

```python
import needle as ndl, needle.nn as nn

# Drop-in: add use_flash_attention=True to any AttentionLayer
layer = nn.AttentionLayer(
    q_features=256, num_head=8, dim_head=32,
    causal=True, use_flash_attention=True,
    device=ndl.cuda()
)
output = layer(x)   # x: (batch, seq_len, 256)

# Or call the op directly (Q,K,V shape: batch, heads, seq_len, head_dim)
from needle.ops import flash_attention
output = flash_attention(Q, K, V, causal=True)
```

```c
// C API
#include "flash_attention.h"
FlashAttentionParams p = {.batch=1, .heads=4, .seq_len=1024, .head_dim=64, .causal=1};
flash_attention_cpu_forward(Q, K, V, O, &p);
```

See [docs/INSTALL.md](docs/INSTALL.md) for CUDA, Python, and optional OpenMP setup.

---

## References

1. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.** *NeurIPS*, 2022.
2. Zico Kolter and David Guestrin. *Deep Learning Systems.* Carnegie Mellon University, 2023. (Needle framework open-source materials)

---

## License and meta

- [LICENSE](LICENSE) — MIT
- [CREDITS](CREDITS)
- [HISTORY.md](HISTORY.md)
- [SECURITY.md](SECURITY.md)
