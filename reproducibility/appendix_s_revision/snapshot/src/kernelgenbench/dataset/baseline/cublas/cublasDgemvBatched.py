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
        _cublas_func = libcublas.cublasDgemvBatched
        _cublas_func.argtypes = [
            ctypes.c_void_p,                 # handle
            ctypes.c_int,                    # trans
            ctypes.c_int,                    # m
            ctypes.c_int,                    # n
            ctypes.POINTER(ctypes.c_double), # alpha (device)
            ctypes.POINTER(ctypes.c_void_p), # Aarray (device pointer array)
            ctypes.c_int,                    # lda
            ctypes.POINTER(ctypes.c_void_p), # xarray (device pointer array)
            ctypes.c_int,                    # incx
            ctypes.POINTER(ctypes.c_double), # beta (device)
            ctypes.POINTER(ctypes.c_void_p), # yarray (device pointer array)
            ctypes.c_int,                    # incy
            ctypes.c_int                     # batchCount
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

def cublasDgemvBatched(trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount):
    '''ctypes cuBLAS C API baseline for cublasDgemvBatched'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    if isinstance(trans, str):
        trans = 0 if trans == 'N' else (1 if trans == 'T' else 2)

    # Aarray/xarray/yarray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    xarray_ptr = ctypes.c_void_p(xarray.data_ptr())
    yarray_ptr = ctypes.c_void_p(yarray.data_ptr())

    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(trans),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(lda),
        ctypes.cast(xarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(incx),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(yarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(incy),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasDgemvBatched failed with status {status}")
    return yarray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    batchCount = 3
    m, n = 4, 5
    alpha = 1.7
    beta = 0.3
    incx = 1
    incy = 1

    # Create batched matrices and vectors on GPU (double precision)
    A_list = [torch.randn(m, n, dtype=torch.float64, device='cuda').contiguous() for _ in range(batchCount)]
    x_list = [torch.randn(n, dtype=torch.float64, device='cuda').contiguous() for _ in range(batchCount)]
    y_list = [torch.randn(m, dtype=torch.float64, device='cuda').contiguous() for _ in range(batchCount)]

    A_list_ref = [A.clone() for A in A_list]
    x_list_ref = [x.clone() for x in x_list]
    y_list_orig = [y.clone() for y in y_list]

    # Build device pointer arrays
    Aarray = torch.tensor([A.data_ptr() for A in A_list], dtype=torch.int64, device='cuda')
    xarray = torch.tensor([x.data_ptr() for x in x_list], dtype=torch.int64, device='cuda')
    yarray = torch.tensor([y.data_ptr() for y in y_list], dtype=torch.int64, device='cuda')

    # Column-major trick for gemv:
    # Row-major A(m,n) is read as column-major (n,m) by cuBLAS.
    # We want y = alpha * A * x + beta * y, i.e. (m,n)*(n,) -> (m,)
    # cuBLAS sees A_cm as (n,m). With trans='T': op(A_cm) = A_cm^T = (m,n), then y(m) = alpha*(m,n)*x(n)+beta*y(m) ✓
    # cuBLAS m_param=n (rows of A_cm), n_param=m (cols of A_cm), lda=n
    result_ptrs = cublasDgemvBatched('T', n, m, alpha, Aarray, n, xarray, incx, beta, yarray, incy, batchCount)
    assert result_ptrs is not None

    expected_list = [alpha * (A_list_ref[i] @ x_list_ref[i]) + beta * y_list_orig[i] for i in range(batchCount)]

    result = torch.stack(y_list, dim=0)
    expected = torch.stack(expected_list, dim=0)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDgemvBatched test passed")