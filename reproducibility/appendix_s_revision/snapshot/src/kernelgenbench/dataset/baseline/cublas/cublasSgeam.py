import torch
import ctypes

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

def _get_cublas_lib():
    global _libcublas
    if _libcublas is None:
        _libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')
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
        _cublas_func = libcublas.cublasSgeam
        _cublas_func.argtypes = [
            ctypes.c_void_p,          # handle
            ctypes.c_int,             # transa
            ctypes.c_int,             # transb
            ctypes.c_int,             # m
            ctypes.c_int,             # n
            ctypes.POINTER(ctypes.c_float),  # alpha (device pointer)
            ctypes.c_void_p,          # A (device pointer)
            ctypes.c_int,             # lda
            ctypes.POINTER(ctypes.c_float),  # beta (device pointer)
            ctypes.c_void_p,          # B (device pointer)
            ctypes.c_int,             # ldb
            ctypes.c_void_p,          # C (device pointer)
            ctypes.c_int              # ldc
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

def cublasSgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
    '''ctypes cuBLAS C API baseline for cublasSgeam'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Call cuBLAS C API
    func(
        handle,
        transa,
        transb,
        m,
        n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        A_ptr,
        lda,
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        B_ptr,
        ldb,
        C_ptr,
        ldc
    )

    return C

if __name__ == "__main__":
    # Constants for cublasOperation_t
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1

    # Test data on GPU
    m, n = 64, 32
    A = torch.randn(m, n, dtype=torch.float32, device='cuda')
    B = torch.randn(m, n, dtype=torch.float32, device='cuda')
    alpha = 1.25
    beta = -0.75

    # Clone originals for reference
    A_ref = A.clone()
    B_ref = B.clone()

    # Prepare column-major compatible buffers by using transposed contiguous storage
    # Row-major contiguous of X.t() equals column-major contiguous of X
    A_cm = A.t().contiguous()
    B_cm = B.t().contiguous()
    C_cm = torch.empty(n, m, dtype=torch.float32, device='cuda')  # column-major buffer for C (m x n)

    # Call baseline (N,N) with column-major layout: lda=ldb=ldc=m
    out_cm = cublasSgeam(
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n,
        alpha, A_cm, m,
        beta, B_cm, m,
        C_cm, m
    )

    assert out_cm is not None

    # Convert back to row-major view for comparison
    result = out_cm.t()

    # PyTorch reference in row-major
    expected = alpha * A_ref + beta * B_ref

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSgeam test passed")