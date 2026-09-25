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
        _cublas_func = libcublas.cublasSsyrk_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p,            # handle
            ctypes.c_int,               # uplo
            ctypes.c_int,               # trans
            ctypes.c_int,               # n
            ctypes.c_int,               # k
            ctypes.POINTER(ctypes.c_float),  # alpha
            ctypes.c_void_p,            # A
            ctypes.c_int,               # lda
            ctypes.POINTER(ctypes.c_float),  # beta
            ctypes.c_void_p,            # C
            ctypes.c_int                # ldc
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

def cublasSsyrk_v2(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    '''ctypes cuBLAS C API baseline for cublasSsyrk_v2'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Call cuBLAS C API
    status = func(
        handle,
        uplo,
        trans,
        n,
        k,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        A_ptr,
        lda,
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        C_ptr,
        ldc
    )
    if status != 0:
        raise RuntimeError(f"cublasSsyrk_v2 failed with status {status}")

    return C

if __name__ == "__main__":
    # Constants for cuBLAS enums
    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_FILL_MODE_UPPER = 1
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1
    CUBLAS_OP_C = 2

    # Test data
    torch.manual_seed(0)
    n = 5
    k = 3
    alpha = 0.75
    beta = 0.25

    A = torch.randn(n, k, dtype=torch.float32, device='cuda').contiguous()
    C = torch.randn(n, n, dtype=torch.float32, device='cuda').contiguous()
    C_orig = C.clone()

    # Parameters adjusted for column-major expectations:
    # Using trans = T so that cuBLAS computes A^T * A with A treated as column-major (k x n)
    uplo = CUBLAS_FILL_MODE_LOWER  # cuBLAS lower triangle corresponds to upper triangle in row-major
    trans = CUBLAS_OP_T
    lda = k  # leading dimension when trans==T (rows of A in column-major)
    ldc = n  # leading dimension for C (rows in column-major)

    # Call baseline
    result = cublasSsyrk_v2(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    assert result is not None

    # PyTorch reference (row-major): update upper triangular of C with alpha*A@A.T + beta*C
    S_full = alpha * (A @ A.t()) + beta * C_orig
    expected = C_orig.clone()
    upper_mask = torch.ones(n, n, dtype=torch.bool, device='cuda').triu()
    expected[upper_mask] = S_full[upper_mask]

    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
    print("✓ cublasSsyrk_v2 test passed")