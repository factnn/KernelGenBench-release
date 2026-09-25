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
        _cublas_func = libcublas.cublasZswap_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # Cache key must include the scalar value
    if dtype in (torch.complex64, torch.complex128):
        cache_key = (key, dtype, complex(value))
    else:
        cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasZswap_v2(n, x, incx, y, incy):
    '''ctypes cuBLAS C API baseline for cublasZswap_v2'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Call cuBLAS C API
    status = func(handle, n, x_ptr, incx, y_ptr, incy)
    if status != 0:
        raise RuntimeError(f"cublasZswap_v2 failed with status {status}")

    return x

if __name__ == "__main__":
    # Test code
    n = 8
    incx = 1
    incy = 1

    xr = torch.randn(n, device='cuda', dtype=torch.float64)
    xi = torch.randn(n, device='cuda', dtype=torch.float64)
    yr = torch.randn(n, device='cuda', dtype=torch.float64)
    yi = torch.randn(n, device='cuda', dtype=torch.float64)

    x = xr + 1j * xi
    y = yr + 1j * yi

    x0 = x.clone()
    y0 = y.clone()

    result = cublasZswap_v2(n, x, incx, y, incy)
    assert result is not None

    expected_x = y0.clone()
    expected_y = x0.clone()

    torch.testing.assert_close(x, expected_x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y, expected_y, rtol=1e-5, atol=1e-5)
    print("✓ cublasZswap_v2 test passed")