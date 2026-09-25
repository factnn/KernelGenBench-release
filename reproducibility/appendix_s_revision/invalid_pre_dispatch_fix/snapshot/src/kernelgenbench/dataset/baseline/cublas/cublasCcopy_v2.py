import torch
import ctypes
import os

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

# Define cuComplex struct for typed pointer casting
class _cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

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
        _cublas_func = libcublas.cublasCcopy_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p,            # handle
            ctypes.c_int,               # n
            ctypes.POINTER(_cuComplex), # x
            ctypes.c_int,               # incx
            ctypes.POINTER(_cuComplex), # y
            ctypes.c_int                # incy
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

def cublasCcopy_v2(n, x, incx, y, incy):
    '''ctypes cuBLAS C API baseline for cublasCcopy_v2'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers and cast to typed pointers
    x_ptr_void = ctypes.c_void_p(x.data_ptr())
    y_ptr_void = ctypes.c_void_p(y.data_ptr())
    x_ptr = ctypes.cast(x_ptr_void, ctypes.POINTER(_cuComplex))
    y_ptr = ctypes.cast(y_ptr_void, ctypes.POINTER(_cuComplex))

    # Call cuBLAS C API
    func(handle, n, x_ptr, incx, y_ptr, incy)

    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'

    # Parameters
    n = 5
    incx = 2
    incy = 3

    # Sizes accounting for increments
    size_x = 1 + (n - 1) * incx
    size_y = 1 + (n - 1) * incy

    # Create test tensors on GPU
    x = torch.randn(size_x, dtype=torch.complex64, device=device) + 1j * torch.randn(size_x, dtype=torch.complex64, device=device)
    y = torch.randn(size_y, dtype=torch.complex64, device=device) + 1j * torch.randn(size_y, dtype=torch.complex64, device=device)

    # Clone originals
    x_clone = x.clone()
    y_clone = y.clone()

    # Call baseline
    result = cublasCcopy_v2(n, x, incx, y, incy)
    assert result is not None

    # PyTorch reference
    expected = y_clone.clone()
    expected_slice = expected[::incy]
    x_slice = x_clone[::incx]
    expected_slice[:n] = x_slice[:n]

    # Numerical check
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCcopy_v2 test passed")