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
        _cublas_func = libcublas.cublasDgemvStridedBatched
        _cublas_func.argtypes = [
            ctypes.c_void_p,                 # handle
            ctypes.c_int,                    # trans
            ctypes.c_int,                    # m
            ctypes.c_int,                    # n
            ctypes.POINTER(ctypes.c_double), # alpha (device pointer)
            ctypes.POINTER(ctypes.c_double), # A (device pointer)
            ctypes.c_int,                    # lda
            ctypes.c_longlong,               # strideA (in elements)
            ctypes.POINTER(ctypes.c_double), # x (device pointer)
            ctypes.c_int,                    # incx
            ctypes.c_longlong,               # stridex (in elements)
            ctypes.POINTER(ctypes.c_double), # beta (device pointer)
            ctypes.POINTER(ctypes.c_double), # y (device pointer)
            ctypes.c_int,                    # incy
            ctypes.c_longlong,               # stridey (in elements)
            ctypes.c_int                     # batchCount
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    '''ctypes cuBLAS C API baseline for cublasDgemvStridedBatched'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert string trans to int if needed (N->0, T->1)
    if isinstance(trans, str):
        trans = 0 if trans == 'N' else 1

    # Convert tensors to GPU pointers (void*) and cast to typed pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    A_typed = ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_double))
    x_typed = ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_double))
    y_typed = ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_double))

    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha', float(alpha), torch.float64)
    beta_gpu = _get_scalar_gpu('beta', float(beta), torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())
    alpha_typed = ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double))
    beta_typed = ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_double))

    # Call cuBLAS C API
    status = func(
        handle,
        ctypes.c_int(trans),
        ctypes.c_int(m),
        ctypes.c_int(n),
        alpha_typed,
        A_typed,
        ctypes.c_int(lda),
        ctypes.c_longlong(int(strideA)),
        x_typed,
        ctypes.c_int(incx),
        ctypes.c_longlong(int(stridex)),
        beta_typed,
        y_typed,
        ctypes.c_int(incy),
        ctypes.c_longlong(int(stridey)),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasDgemvStridedBatched failed with status {status}")

    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required for this test"

    # Parameters
    batchCount = 3
    m = 4
    n = 5
    trans = 0  # CUBLAS_OP_N
    alpha = 1.25
    beta = -0.75

    # Leading dimension and strides (column-major semantics)
    lda = m  # must be >= m
    strideA = lda * n  # number of elements between consecutive matrices
    incx = 1
    incy = 1
    len_x = n  # for trans='N', x length is n
    len_y = m  # for trans='N', y length is m
    stridex = incx * len_x
    stridey = incy * len_y

    # Allocate flat storages matching cuBLAS column-major interpretation
    A = torch.randn(batchCount * strideA, dtype=torch.float64, device='cuda')
    x = torch.randn(batchCount * stridex, dtype=torch.float64, device='cuda')
    y = torch.randn(batchCount * stridey, dtype=torch.float64, device='cuda')

    # Clone originals for expected computation
    A_orig = A.clone()
    x_orig = x.clone()
    y_orig = y.clone()

    # Call baseline
    result = cublasDgemvStridedBatched(
        trans, m, n,
        alpha,
        A, lda, strideA,
        x, incx, stridex,
        beta,
        y, incy, stridey,
        batchCount
    )
    assert result is not None

    # Reference computation using PyTorch with column-major interpretation
    y_expected = y_orig.clone()
    for b in range(batchCount):
        A_b = A_orig.as_strided(size=(m, n), stride=(1, lda), storage_offset=b * strideA)
        x_b = x_orig.as_strided(size=(len_x,), stride=(incx,), storage_offset=b * stridex)
        y_b = y_orig.as_strided(size=(len_y,), stride=(incy,), storage_offset=b * stridey)

        y_ref_b = alpha * (A_b @ x_b) + beta * y_b
        y_expected_view = y_expected.as_strided(size=(len_y,), stride=(incy,), storage_offset=b * stridey)
        y_expected_view.copy_(y_ref_b)

    torch.testing.assert_close(y, y_expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDgemvStridedBatched test passed")