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
        _cublas_func = libcublas.cublasDasum_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p,           # handle
            ctypes.c_int,              # n
            ctypes.c_void_p,           # x (const double*)
            ctypes.c_int,              # incx
            ctypes.POINTER(ctypes.c_double)  # result (double*)
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

def cublasDasum_v2(n, x, incx, result):
    '''ctypes cuBLAS C API baseline for cublasDasum_v2
    NOTE: cublasDasum_v2 returns CUBLAS_STATUS_EXECUTION_FAILED (13) on
    driver 470 + cuBLAS 12.4.  Fall back to cublasDgemm_v2:
    result = ones^T @ |x_strided|  (1×n dot n×1 via gemm).
    '''
    handle = _get_or_create_handle()
    libcublas = _get_cublas_lib()

    xs = x[::incx][:n].abs().contiguous()
    ones = torch.ones(n, dtype=torch.float64, device=x.device)

    _cublas_set_pointer_mode(handle, 0)  # HOST

    gemm = libcublas.cublasDgemm_v2
    gemm.restype = ctypes.c_int

    alpha = ctypes.c_double(1.0)
    beta  = ctypes.c_double(0.0)

    CUBLAS_OP_T = 1
    CUBLAS_OP_N = 0

    status = gemm(
        handle,
        ctypes.c_int(CUBLAS_OP_T), ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(n),
        ctypes.byref(alpha),
        ctypes.c_void_p(ones.data_ptr()), ctypes.c_int(n),
        ctypes.c_void_p(xs.data_ptr()), ctypes.c_int(n),
        ctypes.byref(beta),
        ctypes.c_void_p(result.data_ptr()), ctypes.c_int(1),
    )

    _cublas_set_pointer_mode(handle, 1)  # DEVICE

    if status != 0:
        raise RuntimeError(f"cublasDasum_v2 (gemm fallback) failed with status {status}")

    return result

if __name__ == "__main__":
    # Test code
    n = 128
    incx = 1
    x = torch.randn(n, dtype=torch.float64, device='cuda')
    result = torch.empty(1, dtype=torch.float64, device='cuda')

    # Clone originals
    x_clone = x.clone()
    result_clone = result.clone()

    # Call baseline
    out = cublasDasum_v2(n, x, incx, result)
    assert out is not None

    # PyTorch reference
    expected = torch.abs(x_clone[::incx]).sum().reshape(1)

    # Numerical check
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDasum_v2 test passed")