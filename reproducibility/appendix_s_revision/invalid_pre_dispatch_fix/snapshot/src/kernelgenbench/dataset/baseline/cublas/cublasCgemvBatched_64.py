import torch
import ctypes
import os

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func_cgemv_batched_64 = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

# cuComplex struct (for typed pointer casting)
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

def _get_cublas_func_cgemv_batched_64():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func_cgemv_batched_64
    if _cublas_func_cgemv_batched_64 is None:
        libcublas = _get_cublas_lib()
        func = libcublas.cublasCgemvBatched_64
        func.argtypes = [
            ctypes.c_void_p,                    # handle
            ctypes.c_int,                       # trans
            ctypes.c_longlong,                  # m
            ctypes.c_longlong,                  # n
            ctypes.POINTER(cuComplex),          # alpha (device)
            ctypes.POINTER(ctypes.c_void_p),    # Aarray (device pointer array)
            ctypes.c_longlong,                  # lda
            ctypes.POINTER(ctypes.c_void_p),    # xarray (device pointer array)
            ctypes.c_longlong,                  # incx
            ctypes.POINTER(cuComplex),          # beta (device)
            ctypes.POINTER(ctypes.c_void_p),    # yarray (device pointer array)
            ctypes.c_longlong,                  # incy
            ctypes.c_longlong                   # batchCount
        ]
        func.restype = ctypes.c_int
        _cublas_func_cgemv_batched_64 = func
    return _cublas_func_cgemv_batched_64

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    if dtype in (torch.complex64, torch.complex128):
        cache_key = (key, dtype, complex(value))
    else:
        cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCgemvBatched_64(trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount):
    '''ctypes cuBLAS C API baseline for cublasCgemvBatched_64'''
    handle = _get_or_create_handle()
    func = _get_cublas_func_cgemv_batched_64()

    # Convert trans if string
    if isinstance(trans, str):
        # 'N' -> 0, 'T' -> 1, 'C' -> 2
        trans = 0 if trans == 'N' else (1 if trans == 'T' else 2)

    # Aarray/xarray/yarray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    xarray_ptr = ctypes.c_void_p(xarray.data_ptr())
    yarray_ptr = ctypes.c_void_p(yarray.data_ptr())

    # Get cached scalar GPU tensors for alpha and beta
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(trans),
        ctypes.c_longlong(m),
        ctypes.c_longlong(n),
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuComplex)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_longlong(lda),
        ctypes.cast(xarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_longlong(incx),
        ctypes.cast(beta_ptr, ctypes.POINTER(cuComplex)),
        ctypes.cast(yarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_longlong(incy),
        ctypes.c_longlong(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasCgemvBatched_64 failed with status {status}")

    return yarray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'
    dtype = torch.complex64

    # Problem sizes
    batchCount = 3
    m, n = 4, 5
    lda = m
    incx = 1
    incy = 1
    trans = 'N'

    # Scalars
    alpha = complex(0.75, -0.25)
    beta = complex(-0.1, 0.2)

    # Prepare per-batch tensors (row-major in PyTorch)
    A_list = []
    x_list = []
    y_list = []
    y0_list = []

    for i in range(batchCount):
        A = (torch.randn(m, n, device=device) + 1j * torch.randn(m, n, device=device)).to(dtype)
        x = (torch.randn(n, device=device) + 1j * torch.randn(n, device=device)).to(dtype)
        y = (torch.randn(m, device=device) + 1j * torch.randn(m, device=device)).to(dtype)
        y0 = y.clone()

        A_list.append(A)
        x_list.append(x)
        y_list.append(y)
        y0_list.append(y0)

    # Convert matrices to column-major representation by using A^T row-major
    A_cm_list = [A.t().contiguous() for A in A_list]  # column-major equivalent for cuBLAS

    # Build device arrays of pointers (int64 tensors on GPU)
    A_ptrs = torch.tensor([A_cm.data_ptr() for A_cm in A_cm_list], dtype=torch.int64, device=device)
    x_ptrs = torch.tensor([x.data_ptr() for x in x_list], dtype=torch.int64, device=device)
    y_ptrs = torch.tensor([y.data_ptr() for y in y_list], dtype=torch.int64, device=device)

    # Call baseline
    ret = cublasCgemvBatched_64(trans, m, n, alpha, A_ptrs, lda, x_ptrs, incx, beta, y_ptrs, incy, batchCount)
    assert ret is not None

    # Compute PyTorch reference: y = alpha * A @ x + beta * y0 (row-major)
    expected_list = []
    for i in range(batchCount):
        expected_list.append(alpha * (A_list[i] @ x_list[i]) + beta * y0_list[i])
    expected = torch.stack(expected_list, dim=0)

    # Stack results from y_list for comparison
    result = torch.stack(y_list, dim=0)

    # Numerical check
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCgemvBatched_64 test passed")