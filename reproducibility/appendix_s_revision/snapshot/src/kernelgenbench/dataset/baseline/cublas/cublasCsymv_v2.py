import torch
import ctypes
import os

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

# cuComplex definition for ctypes (matches two float components)
class cuComplex(ctypes.Structure):
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
        _cublas_func = libcublas.cublasCsymv_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p,        # handle
            ctypes.c_int,           # uplo
            ctypes.c_int,           # n
            ctypes.POINTER(cuComplex),  # alpha (device pointer)
            ctypes.c_void_p,        # A (device pointer)
            ctypes.c_int,           # lda
            ctypes.c_void_p,        # x (device pointer)
            ctypes.c_int,           # incx
            ctypes.POINTER(cuComplex),  # beta (device pointer)
            ctypes.c_void_p,        # y (device pointer)
            ctypes.c_int            # incy
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    if dtype.is_complex:
        cache_key = (key, dtype, complex(value))
    else:
        cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCsymv_v2(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    '''ctypes cuBLAS C API baseline for cublasCsymv_v2'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Get cached scalar GPU tensors for alpha and beta (complex64)
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Call cuBLAS C API
    func(
        handle,
        uplo,
        n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuComplex)),
        A_ptr,
        lda,
        x_ptr,
        incx,
        ctypes.cast(beta_ptr, ctypes.POINTER(cuComplex)),
        y_ptr,
        incy
    )

    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    n = 5
    # Create complex64 symmetric matrix A (A == A.t(), not Hermitian)
    M = torch.randn(n, n, device='cuda', dtype=torch.complex64)
    A = 0.5 * (M + M.t())
    x = torch.randn(n, device='cuda', dtype=torch.complex64)
    y = torch.randn(n, device='cuda', dtype=torch.complex64)

    # Clone originals
    A_ref = A.clone()
    x_ref = x.clone()
    y_ref = y.clone()

    alpha = complex(1.25, -0.5)
    beta = complex(-0.3, 0.75)

    # cuBLAS parameters
    CUBLAS_FILL_MODE_UPPER = 0
    uplo = CUBLAS_FILL_MODE_UPPER
    lda = n
    incx = 1
    incy = 1

    # Call baseline
    y_out = cublasCsymv_v2(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    assert y_out is not None

    # PyTorch reference (column-major vs row-major yields A^T, but A is symmetric so equal)
    expected = alpha * (A_ref @ x_ref) + beta * y_ref

    torch.testing.assert_close(y_out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCsymv_v2 test passed")