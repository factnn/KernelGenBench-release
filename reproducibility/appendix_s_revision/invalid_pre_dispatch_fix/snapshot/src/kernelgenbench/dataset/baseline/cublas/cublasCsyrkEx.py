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
        _cublas_func = libcublas.cublasCsyrkEx
        _cublas_func.argtypes = [
            ctypes.c_void_p,               # handle
            ctypes.c_int,                  # uplo
            ctypes.c_int,                  # trans
            ctypes.c_int,                  # n
            ctypes.c_int,                  # k
            ctypes.POINTER(ctypes.c_float),# alpha (cuComplex*)
            ctypes.c_void_p,               # A (const void*)
            ctypes.c_int,                  # Atype (cudaDataType)
            ctypes.c_int,                  # lda
            ctypes.POINTER(ctypes.c_float),# beta (cuComplex*)
            ctypes.c_void_p,               # C (void*)
            ctypes.c_int,                  # Ctype (cudaDataType)
            ctypes.c_int                   # ldc
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    if dtype in (torch.complex64, torch.complex128):
        cache_key = (key, dtype, complex(value))
    else:
        cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCsyrkEx(uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc):
    '''ctypes cuBLAS C API baseline for cublasCsyrkEx'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors (complex64)
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Cast scalar pointers to typed pointers (cuComplex is two floats; typed pointer suffices)
    alpha_typed = ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float))
    beta_typed = ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float))

    # Call cuBLAS C API
    func(handle, uplo, trans, n, k, alpha_typed, A_ptr, Atype, lda, beta_typed, C_ptr, Ctype, ldc)

    return C

if __name__ == "__main__":
    # Constants for enums
    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_OP_N = 0
    CUDA_C_32F = 8

    # Test data
    n = 3
    k = 2
    uplo = CUBLAS_FILL_MODE_LOWER
    trans = CUBLAS_OP_N
    alpha = complex(1.0, 0.5)
    beta = complex(0.0, 0.0)

    # Create input matrices (PyTorch row-major)
    A = torch.randn(n, k, dtype=torch.complex64, device='cuda')
    C = torch.zeros(n, n, dtype=torch.complex64, device='cuda')

    # Clone originals
    A_orig = A.clone()
    C_orig = C.clone()

    # Call baseline
    result = cublasCsyrkEx(uplo, trans, n, k, alpha, A, CUDA_C_32F, n, beta, C, CUDA_C_32F, n)
    assert result is not None

    # PyTorch reference considering column-major expectation of cuBLAS:
    # cuBLAS sees A_cm = A_rm^T and C_cm = C_rm^T due to memory layout mismatch
    A_cm = A_orig.t().contiguous()
    C_cm = C_orig.t().contiguous()

    # Compute R_cm = alpha * A_cm * A_cm^T + beta * C_cm
    R_cm = alpha * (A_cm @ A_cm.transpose(-2, -1)) + beta * C_cm

    # Map back to row-major view: R_rm = R_cm^T
    R_rm = R_cm.transpose(-2, -1).contiguous()

    # Since syrk updates only one triangle (specified by uplo in column-major),
    # and beta=0 with initial C=0, in row-major view:
    # - uplo == LOWER (cm) corresponds to updating UPPER (rm)
    # - uplo == UPPER (cm) corresponds to updating LOWER (rm)
    if uplo == CUBLAS_FILL_MODE_LOWER:
        expected = torch.triu(R_rm)
    else:
        expected = torch.tril(R_rm)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCsyrkEx test passed")