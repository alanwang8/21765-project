# Installation and build (sysadmin)

## Supported platforms

- **Linux x86_64** with GCC or Clang, Python 3.9+, CMake 3.x, and optional **CUDA** (for `ndarray_backend_cuda`).
- **macOS** (Apple Silicon or Intel): CPU backend and **C API** build out of the box; **CUDA** is not available; **OpenMP** is not enabled by default on Apple Clang (see below).

## Dependencies

- CMake, Make, C++17 compiler (for `src/*.cc`), NVCC if building CUDA.
- Python 3 with `pip install pybind11` (or use a virtual environment).
- For running Python tests: `pytest`, `numpy`, and other packages imported by the test suite.

## Building Needle backends

From the repository root:

```bash
make clean   # optional
make lib
```

This configures CMake in `build/`, compiles `ndarray_backend_cpu` and (if CUDA is found) `ndarray_backend_cuda`, and places shared libraries under `python/needle/backend_ndarray/`.

## CPU C reference (`c_api/`)

No CUDA required:

```bash
make c-api
make test-c
make example-c    # runs examples/bench_flash.c
```

## Tests

- **C reference:** `make test-c` — compares Flash-style CPU forward to a naive softmax implementation.
- **Python:** run `pytest` from the repo root after installing dev dependencies (see `tests/`).

## Optional OpenMP (`make test-c-omp`)

Parallelism uses `#pragma omp parallel for` over batch and head dimensions in `c_api/src/flash_attention.c` when compiled with `-DUSE_OPENMP -fopenmp`.

- **GCC / Linux:** usually `make test-c-omp` works as-is.
- **Apple Clang:** often lacks `-fopenmp`; install **LLVM** or **GCC** (e.g. Homebrew `gcc` / `libomp`) and run `make CC=gcc-14 test-c-omp` (adjust compiler name).

## Security note

See [SECURITY.md](../SECURITY.md) for resource limits and numerical considerations.
