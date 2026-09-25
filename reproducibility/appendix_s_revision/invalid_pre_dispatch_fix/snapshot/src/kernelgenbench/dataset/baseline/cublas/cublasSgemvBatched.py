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
        _cublas_func = libcublas.cublasSgemvBatched
        _cublas_func.argtypes = [
            ctypes.c_void_p,                    # handle
            ctypes.c_int,                       # trans
            ctypes.c_int,                       # m
            ctypes.c_int,                       # n
            ctypes.POINTER(ctypes.c_float),     # alpha (device)
            ctypes.POINTER(ctypes.c_void_p),    # Aarray (device pointer array)
            ctypes.c_int,                       # lda
            ctypes.POINTER(ctypes.c_void_p),    # xarray (device pointer array)
            ctypes.c_int,                       # incx
            ctypes.POINTER(ctypes.c_float),     # beta (device)
            ctypes.POINTER(ctypes.c_void_p),    # yarray (device pointer array)
            ctypes.c_int,                       # incy
            ctypes.c_int                        # batchCount
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

def cublasSgemvBatched(trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount):
    '''ctypes cuBLAS C API baseline for cublasSgemvBatched'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Map trans if provided as string
    if isinstance(trans, str):
        trans = 0 if trans == 'N' else (1 if trans == 'T' else 2)

    # Aarray/xarray/yarray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    xarray_ptr = ctypes.c_void_p(xarray.data_ptr())
    yarray_ptr = ctypes.c_void_p(yarray.data_ptr())

    # Get cached scalar GPU tensors for alpha and beta
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(trans),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(lda),
        ctypes.cast(xarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(incx),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(yarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(incy),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasSgemvBatched failed with status {status}")

    return yarray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    m, n = 4, 5
    batchCount = 3
    lda = m
    incx = 1
    incy = 1
    alpha = 0.75
    beta = 0.25

    # Create per-batch matrices/vectors on GPU
    A_buf_list = []   # column-major buffers (store A_desired in column-major by using A_desired.t().contiguous())
    x_list = []
    y_list = []
    y_orig_list = []
    for i in range(batchCount):
        A_desired = torch.randn(m, n, device='cuda', dtype=torch.float32)
        A_buf = A_desired.t().contiguous()  # row-major buffer whose memory matches column-major of A_desired
        x = torch.randn(n, device='cuda', dtype=torch.float32)
        y = torch.randn(m, device='cuda', dtype=torch.float32)
        y_orig = y.clone()

        A_buf_list.append(A_buf)
        x_list.append(x)
        y_list.append(y)
        y_orig_list.append(y_orig)

    # Build device arrays of pointers (int64 tensors on GPU)
    A_ptrs = torch.tensor([t.data_ptr() for t in A_buf_list], dtype=torch.int64, device='cuda')
    X_ptrs = torch.tensor([t.data_ptr() for t in x_list], dtype=torch.int64, device='cuda')
    Y_ptrs = torch.tensor([t.data_ptr() for t in y_list], dtype=torch.int64, device='cuda')

    # Call baseline with trans='N' because A_buf encodes A_desired in column-major layout
    result_yarray = cublasSgemvBatched('N', m, n, alpha, A_ptrs, lda, X_ptrs, incx, beta, Y_ptrs, incy, batchCount)
    assert result_yarray is not None

    # Compute expected results using PyTorch (accounting for column-major by transposing A_buf back)
    expected_list = []
    for i in range(batchCount):
        A_desired = A_buf_list[i].t()  # get A_desired back
        expected = alpha * (A_desired @ x_list[i]) + beta * y_orig_list[i]
        expected_list.append(expected)

    # Stack results and expected for comparison
    result = torch.stack(y_list, dim=0)
    expected = torch.stack(expected_list, dim=0)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSgemvBatched test passed")