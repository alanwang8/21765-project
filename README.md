# Needle Flash Attention

Memory-efficient **scaled dot-product attention** (Flash-style online softmax) integrated into the **Needle** ndarray framework: Python `TensorOp` API, CUDA implementation for training, and a small **CPU C reference** under `c_api/` for correctness and teaching.

## Quick links

| Audience | Document |
|----------|----------|
| Build and platforms | [docs/INSTALL.md](docs/INSTALL.md) |
| Using Flash attention from Python | [docs/USER.md](docs/USER.md) |
| Algorithms and code map | [docs/DEVELOPER.md](docs/DEVELOPER.md) |
| Project overview (HTML) | [www/index.html](www/index.html) |
| Benchmarks and report (Jupyter) | [Flash_Attention_Final_Report.ipynb](Flash_Attention_Final_Report.ipynb) |
| Man page (library API) | [man/flash_attention.3](man/flash_attention.3) (`man ./man/flash_attention.3` from repo root) |

## Build

```bash
make          # CMake backends + C API object
make test-c   # CPU reference vs naive attention
make lib      # Needle ndarray CPU/CUDA only
```

See [docs/INSTALL.md](docs/INSTALL.md) for CUDA, Python, and optional OpenMP.

## License and meta

- [LICENSE](LICENSE) — MIT
- [CREDITS](CREDITS)
- [HISTORY.md](HISTORY.md)
- [SECURITY.md](SECURITY.md)
