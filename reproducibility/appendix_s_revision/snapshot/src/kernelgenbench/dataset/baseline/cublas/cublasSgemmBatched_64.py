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
        status = _cublas_set_pointer_mode(_cublas_handle, 1)
        if status != 0:
            raise RuntimeError(f"cublasSetPointerMode_v2 failed with status {status}")

    return _cublas_handle

def _get_cublas_func():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasSgemmBatched_64
        _cublas_func.argtypes = [
            ctypes.c_void_p,                  # handle
            ctypes.c_int,                     # transa
            ctypes.c_int,                     # transb
            ctypes.c_int64,                   # m
            ctypes.c_int64,                   # n
            ctypes.c_int64,                   # k
            ctypes.POINTER(ctypes.c_float),   # alpha (device)
            ctypes.POINTER(ctypes.c_void_p),  # Aarray (device pointer array)
            ctypes.c_int64,                   # lda
            ctypes.POINTER(ctypes.c_void_p),  # Barray (device pointer array)
            ctypes.c_int64,                   # ldb
            ctypes.POINTER(ctypes.c_float),   # beta (device)
            ctypes.POINTER(ctypes.c_void_p),  # Carray (device pointer array)
            ctypes.c_int64,                   # ldc
            ctypes.c_int64                    # batchCount
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

def cublasSgemmBatched_64(transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount):
    '''ctypes cuBLAS C API baseline for cublasSgemmBatched_64'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Optional conversion if transa/transb are provided as characters
    if isinstance(transa, str):
        transa = 0 if transa == 'N' else (1 if transa == 'T' else 2)
    if isinstance(transb, str):
        transb = 0 if transb == 'N' else (1 if transb == 'T' else 2)

    # Aarray/Barray/Carray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    Barray_ptr = ctypes.c_void_p(Barray.data_ptr())
    Carray_ptr = ctypes.c_void_p(Carray.data_ptr())

    # Get cached scalar GPU tensors for alpha/beta
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa), ctypes.c_int(transb),
        ctypes.c_int64(m), ctypes.c_int64(n), ctypes.c_int64(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int64(lda),
        ctypes.cast(Barray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int64(ldb),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(Carray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int64(ldc),
        ctypes.c_int64(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasSgemmBatched_64 failed with status {status}")
    return Carray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'

    # Dimensions
    m, n, k = 4, 5, 6
    batchCount = 3

    # Create batched input matrices on GPU (row-major)
    A_list = [torch.randn(m, k, dtype=torch.float32, device=device) for _ in range(batchCount)]
    B_list = [torch.randn(k, n, dtype=torch.float32, device=device) for _ in range(batchCount)]
    C_list = [torch.randn(m, n, dtype=torch.float32, device=device) for _ in range(batchCount)]

    # Clones for reference computation (row-major)
    A_ref = [a.clone() for a in A_list]
    B_ref = [b.clone() for b in B_list]
    C_ref = [c.clone() for c in C_list]

    # Build column-major representations by transposing and making contiguous
    # Row-major (m,k) -> Column-major (m,k) by using a.t().contiguous() with shape (k,m)
    A_cm = [a.t().contiguous() for a in A_list]  # treated as column-major (m,k)
    B_cm = [b.t().contiguous() for b in B_list]  # treated as column-major (k,n)
    C_cm = [c.t().contiguous() for c in C_list]  # treated as column-major (m,n)

    # Create device arrays of pointers (int64) to each matrix
    Aarray = torch.tensor([int(x.data_ptr()) for x in A_cm], dtype=torch.int64, device=device)
    Barray = torch.tensor([int(x.data_ptr()) for x in B_cm], dtype=torch.int64, device=device)
    Carray = torch.tensor([int(x.data_ptr()) for x in C_cm], dtype=torch.int64, device=device)

    # Scalars
    alpha = 1.5
    beta = -0.25

    # Leading dimensions for column-major: lda=m, ldb=k, ldc=m
    lda = m
    ldb = k
    ldc = m

    # Call baseline
    out = cublasSgemmBatched_64('N', 'N', m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    assert out is not None

    # Compute expected result in PyTorch (row-major)
    expected_list = []
    for i in range(batchCount):
        expected = alpha * (A_ref[i] @ B_ref[i]) + beta * C_ref[i]
        expected_list.append(expected)

    # Retrieve result from column-major buffers by transposing back
    result_list = [C_cm[i].t().contiguous() for i in range(batchCount)]

    # Numerical check
    for i in range(batchCount):
        torch.testing.assert_close(result_list[i], expected_list[i], rtol=1e-2, atol=1e-2)

    print("✓ cublasSgemmBatched_64 test passed")