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
        _cublas_func = libcublas.cublasDsbmv_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p,          # handle
            ctypes.c_int,             # uplo
            ctypes.c_int,             # n
            ctypes.c_int,             # k
            ctypes.POINTER(ctypes.c_double),  # alpha (device pointer)
            ctypes.c_void_p,          # A (device pointer)
            ctypes.c_int,             # lda
            ctypes.c_void_p,          # x (device pointer)
            ctypes.c_int,             # incx
            ctypes.POINTER(ctypes.c_double),  # beta (device pointer)
            ctypes.c_void_p,          # y (device pointer)
            ctypes.c_int              # incy
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDsbmv_v2(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    '''ctypes cuBLAS C API baseline for cublasDsbmv_v2'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float64)
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Call cuBLAS C API
    func(
        handle,
        uplo,
        n,
        k,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        A_ptr,
        lda,
        x_ptr,
        incx,
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_double)),
        y_ptr,
        incy
    )

    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    n = 8
    k = 2
    lda = k + 1
    # cublasFillMode_t: 1 for upper, 0 for lower (cuBLAS convention)
    uplo = 1  # test upper storage

    # Create a symmetric band matrix in full form
    A_full = torch.zeros((n, n), dtype=torch.float64, device='cuda')
    for d in range(k + 1):
        vals = torch.randn(n - d, dtype=torch.float64, device='cuda')
        for i in range(n - d):
            A_full[i, i + d] = vals[i]
            A_full[i + d, i] = vals[i]  # ensure symmetry

    # Build band storage AB with shape (lda, n) according to BLAS conventions
    AB = torch.zeros((lda, n), dtype=torch.float64, device='cuda')
    if uplo == 1:
        # Upper storage: AB[k + i - j, j] = A_full[i, j] for i in [max(0, j-k), j]
        for j in range(n):
            i_start = max(0, j - k)
            for i in range(i_start, j + 1):
                row = k + i - j
                AB[row, j] = A_full[i, j]
    else:
        # Lower storage: AB[i - j, j] = A_full[i, j] for i in [j, min(n-1, j+k)]
        for j in range(n):
            i_end = min(n - 1, j + k)
            for i in range(j, i_end + 1):
                row = i - j
                AB[row, j] = A_full[i, j]

    # Prepare matrix for cuBLAS (column-major): use transpose trick
    A_band_for_cublas = AB.t().contiguous()

    # Vectors x and y
    x = torch.randn(n, dtype=torch.float64, device='cuda')
    y = torch.randn(n, dtype=torch.float64, device='cuda')
    y_orig = y.clone()

    alpha = 1.25
    beta = -0.75
    incx = 1
    incy = 1

    # Call baseline
    y_out = cublasDsbmv_v2(uplo, n, k, alpha, A_band_for_cublas, lda, x, incx, beta, y, incy)
    assert y_out is not None

    # Reference computation using full matrix
    y_expected = alpha * (A_full @ x) + beta * y_orig

    torch.testing.assert_close(y_out, y_expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDsbmv_v2 test passed")