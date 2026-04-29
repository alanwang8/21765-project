# Project history

## 2025-04 — Public release packaging

- Added `c_api/` CPU reference (`flash_attention_cpu_forward`) with `make test-c`.
- Added multi-audience docs (`docs/INSTALL.md`, `USER.md`, `DEVELOPER.md`, `index.html`), meta files (`LICENSE`, `CREDITS`, `SECURITY.md`).
- Extended root `Makefile` with C targets: `c-api`, `test-c`, `example-c`, optional OpenMP variants.

## Earlier milestones

- Integrated Flash attention forward and backward into CUDA ndarray backend and `ops_attention.py`.
- Benchmarked memory and latency against PyTorch naive and SDPA paths.
