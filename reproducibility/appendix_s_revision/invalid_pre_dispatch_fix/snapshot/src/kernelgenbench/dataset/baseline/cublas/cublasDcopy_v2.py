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

        cublasCreate_v2 = libcublas.cublasCreate_v2
        cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cublasCreate_v2.restype = ctypes.c_int
        _cublas_handle = ctypes.c_void_p()
        status = cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")

        _cublas_set_pointer_mode = libcublas.cublasSetPointerMode_v2
        _cublas_set_pointer_mode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _cublas_set_pointer_mode.restype = ctypes.c_int

        _cublas_set_pointer_mode(_cublas_handle, 1)

    return _cublas_handle

def _get_cublas_func():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasDcopy_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p,                 # handle
            ctypes.c_int,                    # n
            ctypes.POINTER(ctypes.c_double), # x
            ctypes.c_int,                    # incx
            ctypes.POINTER(ctypes.c_double), # y
            ctypes.c_int                     # incy
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def cublasDcopy_v2(n, x, incx, y, incy):
    '''ctypes cuBLAS C API baseline for cublasDcopy_v2: y = x'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    func(
        handle, n,
        ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_double)), incx,
        ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_double)), incy
    )

    return y

if __name__ == "__main__":
    torch.manual_seed(0)
    n = 100
    x = torch.randn(n, dtype=torch.float64, device='cuda')
    y = torch.randn(n, dtype=torch.float64, device='cuda')

    result = cublasDcopy_v2(n, x, 1, y, 1)
    torch.testing.assert_close(result, x, rtol=1e-7, atol=1e-7)
    print("✓ cublasDcopy_v2 test passed")
