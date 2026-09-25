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
        _cublas_func = libcublas.cublasSdot_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p
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

def cublasSdot_v2(n, x, incx, y, incy, result):
    '''ctypes cuBLAS C API baseline for cublasSdot_v2
    NOTE: cublasSdot_v2 returns CUBLAS_STATUS_NOT_SUPPORTED (15) on
    driver 470 + cuBLAS 12.4.  Fall back to cublasSgemm_v2 which
    computes the same dot product as  C(1×1) = x^T · y.
    For non-unit strides the strided elements are first gathered into
    contiguous buffers so that lda/ldb >= k.
    '''
    handle = _get_or_create_handle()
    libcublas = _get_cublas_lib()

    # --- gather contiguous vectors --------------------------------
    xs = x[::incx][:n].contiguous()
    ys = y[::incy][:n].contiguous()

    # --- switch to HOST pointer mode for gemm alpha/beta ----------
    _cublas_set_pointer_mode(handle, 0)  # HOST

    # --- cublasSgemm_v2: C(1×1) = alpha * A^T(1×n) * B(n×1) -----
    gemm = libcublas.cublasSgemm_v2
    gemm.restype = ctypes.c_int

    alpha = ctypes.c_float(1.0)
    beta  = ctypes.c_float(0.0)

    CUBLAS_OP_T = 1
    CUBLAS_OP_N = 0

    status = gemm(
        handle,
        ctypes.c_int(CUBLAS_OP_T), ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(n),
        ctypes.byref(alpha),
        ctypes.c_void_p(xs.data_ptr()), ctypes.c_int(n),
        ctypes.c_void_p(ys.data_ptr()), ctypes.c_int(n),
        ctypes.byref(beta),
        ctypes.c_void_p(result.data_ptr()), ctypes.c_int(1),
    )

    # --- restore DEVICE pointer mode ------------------------------
    _cublas_set_pointer_mode(handle, 1)  # DEVICE

    if status != 0:
        raise RuntimeError(f"cublasSdot_v2 (gemm fallback) failed with status {status}")

    return result

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    n = 16
    incx = 2
    incy = 3
    len_x = 1 + (n - 1) * abs(incx)
    len_y = 1 + (n - 1) * abs(incy)

    x = torch.randn(len_x, dtype=torch.float32, device='cuda')
    y = torch.randn(len_y, dtype=torch.float32, device='cuda')
    result = torch.zeros(1, dtype=torch.float32, device='cuda')

    # Clone originals
    x_clone = x.clone()
    y_clone = y.clone()

    # Call baseline
    out = cublasSdot_v2(n, x, incx, y, incy, result)
    assert out is not None

    # PyTorch reference with strides
    xs = x_clone[::incx][:n]
    ys = y_clone[::incy][:n]
    expected = (xs * ys).sum().reshape(1)

    # Numerical check
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSdot_v2 test passed")