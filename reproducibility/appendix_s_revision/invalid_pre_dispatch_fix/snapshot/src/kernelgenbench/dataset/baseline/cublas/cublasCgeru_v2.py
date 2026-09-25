import torch
import ctypes
import os

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

# Define cuComplex struct for ctypes (matches CUDA's cuComplex: two float32)
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
        _cublas_func = libcublas.cublasCgeru_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p,          # handle
            ctypes.c_int,             # m
            ctypes.c_int,             # n
            ctypes.POINTER(cuComplex),# alpha (device pointer)
            ctypes.c_void_p,          # x (device pointer)
            ctypes.c_int,             # incx
            ctypes.c_void_p,          # y (device pointer)
            ctypes.c_int,             # incy
            ctypes.c_void_p,          # A (device pointer)
            ctypes.c_int              # lda
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

def cublasCgeru_v2(m, n, alpha, x, incx, y, incy, A, lda):
    '''ctypes cuBLAS C API baseline for cublasCgeru_v2'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())
    A_ptr = ctypes.c_void_p(A.data_ptr())

    # Get cached scalar GPU tensor for complex alpha
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS C API
    status = func(
        handle,
        m,
        n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuComplex)),
        x_ptr,
        incx,
        y_ptr,
        incy,
        A_ptr,
        lda
    )
    if status != 0:
        raise RuntimeError(f"cublasCgeru_v2 failed with status {status}")

    return A

if __name__ == "__main__":
    # Test code
    torch.manual_seed(123)

    # Dimensions
    m, n = 4, 3
    incx, incy = 1, 1
    lda = m

    # Create A with column-major memory layout by transposing a contiguous (n, m) tensor
    A_base = torch.randn(n, m, dtype=torch.complex64, device='cuda')
    A = A_base.t()  # shape (m, n), column-major in memory

    # Create vectors x (length m) and y (length n)
    x = torch.randn(m, dtype=torch.complex64, device='cuda')
    y = torch.randn(n, dtype=torch.complex64, device='cuda')

    # Scalar alpha (complex64)
    alpha = complex(0.5, -0.3)

    # Clone originals for reference
    A_before = A.clone()
    x_clone = x.clone()
    y_clone = y.clone()

    # Call baseline
    A_out = cublasCgeru_v2(m, n, alpha, x, incx, y, incy, A, lda)
    assert A_out is not None

    # PyTorch reference (unconjugated rank-1 update): A := alpha * x * y^T + A
    expected = A_before + (alpha * x_clone.unsqueeze(1) * y_clone.unsqueeze(0))

    # Numerical check
    torch.testing.assert_close(A_out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCgeru_v2 test passed")