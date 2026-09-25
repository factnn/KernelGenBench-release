import torch
import ctypes
import os

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

def _get_cublas_lib():
    global _libcublas
    if _libcublas is None:
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        _libcublas = ctypes.CDLL(os.path.join(cuda_home, 'lib64', 'libcublas.so.12'))
    return _libcublas

def _get_or_create_handle():
    '''Get or create global cuBLAS handle (reused across calls)'''
    global _cublas_handle, _cublas_set_pointer_mode
    if _cublas_handle is None:
        libcublas = _get_cublas_lib()

        # Create handle
        cublasCreate_v2 = libcublas.cublasCreate_v2
        cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cublasCreate_v2.restype = ctypes.c_int
        _cublas_handle = ctypes.c_void_p()
        status = cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")

        # Setup SetPointerMode function (once)
        _cublas_set_pointer_mode = libcublas.cublasSetPointerMode_v2
        _cublas_set_pointer_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _cublas_set_pointer_mode.restype = ctypes.c_int

        # Set to device mode (once)
        _cublas_set_pointer_mode(_cublas_handle, 1)

    return _cublas_handle

def _get_cublas_func():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasDgemmBatched
        _cublas_func.argtypes = [
            ctypes.c_void_p,                 # handle
            ctypes.c_int,                    # transa
            ctypes.c_int,                    # transb
            ctypes.c_int,                    # m
            ctypes.c_int,                    # n
            ctypes.c_int,                    # k
            ctypes.POINTER(ctypes.c_double), # alpha (device)
            ctypes.POINTER(ctypes.c_void_p), # Aarray (device pointer array)
            ctypes.c_int,                    # lda
            ctypes.POINTER(ctypes.c_void_p), # Barray (device pointer array)
            ctypes.c_int,                    # ldb
            ctypes.POINTER(ctypes.c_double), # beta (device)
            ctypes.POINTER(ctypes.c_void_p), # Carray (device pointer array)
            ctypes.c_int,                    # ldc
            ctypes.c_int                     # batchCount
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDgemmBatched(transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount):
    '''ctypes cuBLAS C API baseline for cublasDgemmBatched'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    if isinstance(transa, str):
        transa = 0 if transa == 'N' else (1 if transa == 'T' else 2)
    if isinstance(transb, str):
        transb = 0 if transb == 'N' else (1 if transb == 'T' else 2)

    # Aarray/Barray/Carray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    Barray_ptr = ctypes.c_void_p(Barray.data_ptr())
    Carray_ptr = ctypes.c_void_p(Carray.data_ptr())

    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa), ctypes.c_int(transb),
        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int(lda),
        ctypes.cast(Barray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int(ldb),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Carray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int(ldc),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasDgemmBatched failed with status {status}")
    return Carray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    batchCount = 3
    m, n, k = 4, 5, 6
    alpha = 0.75
    beta = 1.25

    # Create row-major input matrices on GPU
    A_mats = [torch.randn(m, k, dtype=torch.float64, device='cuda') for _ in range(batchCount)]
    B_mats = [torch.randn(k, n, dtype=torch.float64, device='cuda') for _ in range(batchCount)]
    C_mats = [torch.randn(m, n, dtype=torch.float64, device='cuda') for _ in range(batchCount)]

    # Clone originals for reference
    A_orig = [a.clone() for a in A_mats]
    B_orig = [b.clone() for b in B_mats]
    C_orig = [c.clone() for c in C_mats]

    # Column-major trick (same as cublasSgemm_v2):
    # Row-major C(m,n) = alpha * A(m,k) @ B(k,n) + beta * C(m,n)
    # is equivalent to column-major:
    # C^T(n,m) = alpha * B^T(n,k) @ A^T(k,m) + beta * C^T(n,m)
    # Pass row-major tensors directly: cuBLAS reads them as transposed column-major
    # m'=n, n'=m, k'=k, A_operand=B, B_operand=A, lda=n, ldb=k, ldc=n
    Aarray = torch.tensor([b.data_ptr() for b in B_mats], dtype=torch.int64, device='cuda')
    Barray = torch.tensor([a.data_ptr() for a in A_mats], dtype=torch.int64, device='cuda')
    Carray = torch.tensor([c.data_ptr() for c in C_mats], dtype=torch.int64, device='cuda')

    result_ptrs = cublasDgemmBatched('N', 'N', n, m, k, alpha, Aarray, n, Barray, k, beta, Carray, n, batchCount)
    assert result_ptrs is not None

    expected = [alpha * A_orig[i] @ B_orig[i] + beta * C_orig[i] for i in range(batchCount)]

    for i in range(batchCount):
        torch.testing.assert_close(C_mats[i], expected[i], rtol=1e-5, atol=1e-5)

    print("✓ cublasDgemmBatched test passed")