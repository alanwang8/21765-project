.PHONY: lib c-api c-api-omp test-c test-c-omp example-c example-c-omp test all help clean clean-c format

CC ?= cc
CFLAGS = -O2 -std=c11 -Wall -Wextra -I c_api/include
LDFLAGS = -lm
C_OBJ = c_api/build/flash_attention.o
C_OBJ_OMP = c_api/build/flash_attention_omp.o

all: lib c-api

help:
	@echo "Targets:"
	@echo "  lib          - CMake build of Needle ndarray backends (CPU/CUDA)"
	@echo "  c-api        - CPU reference flash_attention C object"
	@echo "  c-api-omp    - Same with OpenMP (requires compiler OpenMP support)"
	@echo "  test-c       - Correctness test vs naive attention"
	@echo "  test-c-omp   - test-c with OpenMP build"
	@echo "  example-c    - Run examples/bench_flash.c"
	@echo "  test         - Alias for test-c"
	@echo "  clean        - Remove build artifacts"
	@echo "  clean-c      - Remove c_api/build only"

lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

c-api: $(C_OBJ)

$(C_OBJ): c_api/src/flash_attention.c c_api/include/flash_attention.h
	@mkdir -p c_api/build
	$(CC) $(CFLAGS) -c c_api/src/flash_attention.c -o $@

c-api-omp: $(C_OBJ_OMP)

$(C_OBJ_OMP): c_api/src/flash_attention.c c_api/include/flash_attention.h
	@mkdir -p c_api/build
	$(CC) $(CFLAGS) -DUSE_OPENMP -fopenmp -c c_api/src/flash_attention.c -o $@

test-c: $(C_OBJ)
	@mkdir -p c_api/build
	$(CC) $(CFLAGS) c_api/tests/test_flash_attention.c $(C_OBJ) -o c_api/build/test_flash_attention $(LDFLAGS)
	./c_api/build/test_flash_attention

test-c-omp: $(C_OBJ_OMP)
	@mkdir -p c_api/build
	$(CC) $(CFLAGS) -fopenmp c_api/tests/test_flash_attention.c $(C_OBJ_OMP) -o c_api/build/test_flash_attention_omp $(LDFLAGS)
	./c_api/build/test_flash_attention_omp

example-c: $(C_OBJ)
	@mkdir -p c_api/build
	$(CC) $(CFLAGS) examples/bench_flash.c $(C_OBJ) -o c_api/build/bench_flash $(LDFLAGS)
	./c_api/build/bench_flash

example-c-omp: $(C_OBJ_OMP)
	@mkdir -p c_api/build
	$(CC) $(CFLAGS) -fopenmp examples/bench_flash.c $(C_OBJ_OMP) -o c_api/build/bench_flash_omp $(LDFLAGS)
	./c_api/build/bench_flash_omp

test: test-c

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean-c:
	rm -rf c_api/build

clean: clean-c
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so
