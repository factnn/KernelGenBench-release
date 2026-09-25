import torch
import ctypes
import os
import atexit

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
        status = _cublas_set_pointer_mode(_cublas_handle, 1)
        if status != 0:
            raise RuntimeError(f"cublasSetPointerMode_v2 failed with status {status}")

    return _cublas_handle


def cleanup_cublas():
    '''Release cuBLAS handle on exit'''
    global _cublas_handle
    if _cublas_handle is not None:
        libcublas = _get_cublas_lib()
        libcublas.cublasDestroy_v2(_cublas_handle)
        _cublas_handle = None

atexit.register(cleanup_cublas)


def _get_cublas_func():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasCgemvStridedBatched
        _cublas_func.argtypes = [
            ctypes.c_void_p,                 # handle
            ctypes.c_int,                    # trans
            ctypes.c_int,                    # m
            ctypes.c_int,                    # n
            ctypes.POINTER(ctypes.c_float),  # alpha (device pointer to cuComplex)
            ctypes.POINTER(ctypes.c_float),  # A (device pointer to cuComplex)
            ctypes.c_int,                    # lda
            ctypes.c_longlong,               # strideA
            ctypes.POINTER(ctypes.c_float),  # x (device pointer to cuComplex)
            ctypes.c_int,                    # incx
            ctypes.c_longlong,               # stridex
            ctypes.POINTER(ctypes.c_float),  # beta (device pointer to cuComplex)
            ctypes.POINTER(ctypes.c_float),  # y (device pointer to cuComplex)
            ctypes.c_int,                    # incy
            ctypes.c_longlong,               # stridey
            ctypes.c_int                     # batchCount
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, complex(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    '''ctypes cuBLAS C API baseline for cublasCgemvStridedBatched: batched complex64 GEMV with strided storage'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert string trans to int if needed (N->0, T->1)
    if isinstance(trans, str):
        trans = 0 if trans == 'N' else 1
    
    # Map trans to cublasOperation_t
    def _to_cublas_op(t):
        if isinstance(t, int):
            return t
        t = str(t).upper()
        if t == 'N':
            return 0
        elif t == 'T':
            return 1
        elif t == 'C':
            return 2
        else:
            raise ValueError("Invalid trans value. Use 'N', 'T', or 'C'.")
    
    trans_op = _to_cublas_op(trans)
    
    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())
    
    # Cast to typed pointers (cuComplex is two float32s)
    A_p = ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float))
    x_p = ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float))
    y_p = ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_float))
    
    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha_complex64', alpha, torch.complex64)
    beta_gpu = _get_scalar_gpu('beta_complex64', beta, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())
    alpha_p = ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float))
    beta_p = ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float))
    
    # Call cuBLAS C API
    status = func(
        handle, trans_op, m, n, alpha_p, A_p, lda, ctypes.c_longlong(strideA),
        x_p, incx, ctypes.c_longlong(stridex), beta_p, y_p, incy, 
        ctypes.c_longlong(stridey), batchCount
    )
    if status != 0:
        raise RuntimeError(f"cublasCgemvStridedBatched failed with status {status}")
    
    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'

    # Dimensions
    m, n = 5, 4
    batchCount = 3

    # Create test tensors on GPU with correct dtype
    # Complex A: shape (batch, m, n)
    A_real = torch.randn(batchCount, m, n, device=device)
    A_imag = torch.randn(batchCount, m, n, device=device)
    A = torch.complex(A_real, A_imag).contiguous()

    # x vector: shape (batch, n) for trans='N'
    x_real = torch.randn(batchCount, n, device=device)
    x_imag = torch.randn(batchCount, n, device=device)
    x = torch.complex(x_real, x_imag).contiguous()

    # y vector: shape (batch, m)
    y_real = torch.randn(batchCount, m, device=device)
    y_imag = torch.randn(batchCount, m, device=device)
    y = torch.complex(y_real, y_imag).contiguous()

    # Clone originals for comparison
    A_orig = A.clone()
    x_orig = x.clone()
    y_orig = y.clone()

    # Transpose A to align row-major with cuBLAS column-major expectation
    # We'll pass A_t (n, m) row-major so cuBLAS sees (m, n) column-major.
    A_t = A_orig.transpose(-1, -2).contiguous()

    # Scalars
    alpha = complex(1.0, 0.5)
    beta = complex(0.5, 0.0)

    # cuBLAS parameters
    trans = 'N'
    lda = m
    strideA = m * n
    stridex = n
    stridey = m
    incx = 1
    incy = 1

    # Call baseline
    result = cublasCgemvStridedBatched(
        trans, m, n, alpha, A_t, lda, strideA,
        x_orig, incx, stridex, beta, y, incy, stridey, batchCount
    )

    assert result is not None

    # PyTorch reference: y = alpha * A @ x + beta * y
    expected = alpha * (A_orig @ x_orig.unsqueeze(-1)).squeeze(-1) + beta * y_orig

    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    print("✓ cublasCgemvStridedBatched test passed")
