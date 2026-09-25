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
        func = libcublas.cublasSgemvStridedBatched
        func.argtypes = [
            ctypes.c_void_p,                  # handle
            ctypes.c_int,                     # trans
            ctypes.c_int,                     # m
            ctypes.c_int,                     # n
            ctypes.POINTER(ctypes.c_float),   # alpha (device pointer)
            ctypes.POINTER(ctypes.c_float),   # A (device pointer)
            ctypes.c_int,                     # lda
            ctypes.c_longlong,                # strideA (elements)
            ctypes.POINTER(ctypes.c_float),   # x (device pointer)
            ctypes.c_int,                     # incx
            ctypes.c_longlong,                # stridex (elements)
            ctypes.POINTER(ctypes.c_float),   # beta (device pointer)
            ctypes.POINTER(ctypes.c_float),   # y (device pointer)
            ctypes.c_int,                     # incy
            ctypes.c_longlong,                # stridey (elements)
            ctypes.c_int                      # batchCount
        ]
        func.restype = ctypes.c_int
        _cublas_func = func
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    '''ctypes cuBLAS C API baseline for cublasSgemvStridedBatched: batched SGEMV with strided inputs'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert string trans to int if needed (N->0, T->1)
    if isinstance(trans, str):
        trans = 0 if trans == 'N' else 1

    # Extract raw device pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Prepare scalar device pointers (cached)
    alpha_gpu = _get_scalar_gpu(('alpha', 'float32'), float(alpha), torch.float32)
    beta_gpu = _get_scalar_gpu(('beta', 'float32'), float(beta), torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Cast to typed pointers
    alpha_p = ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float))
    beta_p = ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float))
    A_p = ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float))
    x_p = ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float))
    y_p = ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_float))

    status = func(
        handle,
        ctypes.c_int(trans),
        ctypes.c_int(m),
        ctypes.c_int(n),
        alpha_p,
        A_p,
        ctypes.c_int(lda),
        ctypes.c_longlong(int(strideA)),
        x_p,
        ctypes.c_int(incx),
        ctypes.c_longlong(int(stridex)),
        beta_p,
        y_p,
        ctypes.c_int(incy),
        ctypes.c_longlong(int(stridey)),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasSgemvStridedBatched failed with status {status}")

    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'

    # Dimensions
    m = 4
    n = 5
    batchCount = 3

    # Scalars
    alpha = 1.5
    beta = -0.75

    # Create row-major A (batch, m, n)
    A_row = torch.randn(batchCount, m, n, dtype=torch.float32, device=device)
    # Prepare A for cuBLAS column-major by taking transpose and making contiguous
    # This makes memory layout compatible with column-major interpretation for (m, n)
    A_cublas = A_row.transpose(1, 2).contiguous()  # shape (batch, n, m)

    # Vectors
    x = torch.randn(batchCount, n, dtype=torch.float32, device=device).contiguous()
    y = torch.randn(batchCount, m, dtype=torch.float32, device=device).contiguous()
    y_orig = y.clone()

    # cuBLAS parameters
    CUBLAS_OP_N = 0
    trans = CUBLAS_OP_N
    lda = m  # leading dimension (rows of A in column-major)
    strideA = lda * n  # number of elements between consecutive A matrices
    incx = 1
    incy = 1
    stridex = n  # since trans == N, x has length n
    stridey = m  # y has length m
    batchCount_param = batchCount

    # Call baseline
    y_out = cublasSgemvStridedBatched(
        trans, m, n, alpha,
        A_cublas, lda, strideA,
        x, incx, stridex,
        beta,
        y, incy, stridey,
        batchCount_param
    )

    assert y_out is not None

    # PyTorch reference (row-major): y = alpha * (A_row @ x) + beta * y_orig
    expected = alpha * torch.bmm(A_row, x.unsqueeze(2)).squeeze(2) + beta * y_orig

    torch.testing.assert_close(y_out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSgemvStridedBatched test passed")