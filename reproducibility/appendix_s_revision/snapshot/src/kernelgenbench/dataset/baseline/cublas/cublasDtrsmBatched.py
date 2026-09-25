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
        _cublas_func = libcublas.cublasDtrsmBatched
        _cublas_func.argtypes = [
            ctypes.c_void_p,                  # handle
            ctypes.c_int,                     # side
            ctypes.c_int,                     # uplo
            ctypes.c_int,                     # trans
            ctypes.c_int,                     # diag
            ctypes.c_int,                     # m
            ctypes.c_int,                     # n
            ctypes.POINTER(ctypes.c_double),  # alpha (device)
            ctypes.POINTER(ctypes.c_void_p),  # Aarray (device pointer array)
            ctypes.c_int,                     # lda
            ctypes.POINTER(ctypes.c_void_p),  # Barray (device pointer array)
            ctypes.c_int,                     # ldb
            ctypes.c_int                      # batchCount
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount):
    '''ctypes cuBLAS C API baseline for cublasDtrsmBatched'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Aarray/Barray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    Barray_ptr = ctypes.c_void_p(Barray.data_ptr())

    # Get cached scalar GPU tensor for alpha
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(side),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(lda),
        ctypes.cast(Barray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(ldb),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasDtrsmBatched failed with status {status}")

    return Barray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    dtype = torch.float64
    device = 'cuda'

    batchCount = 2
    m = 4
    n = 3
    alpha = 0.7

    CUBLAS_SIDE_LEFT = 0
    CUBLAS_FILL_MODE_UPPER = 1
    CUBLAS_OP_N = 0
    CUBLAS_DIAG_NON_UNIT = 0

    side = CUBLAS_SIDE_LEFT
    uplo = CUBLAS_FILL_MODE_UPPER
    trans = CUBLAS_OP_N
    diag = CUBLAS_DIAG_NON_UNIT

    As = []
    Bs_cm = []
    Bs_orig = []
    for i in range(batchCount):
        diag_vals = 1.0 + torch.rand(m, dtype=dtype, device=device)
        A = torch.diag(diag_vals).to(device=device, dtype=dtype)
        B_rm = torch.randn(m, n, dtype=dtype, device=device)

        As.append(A)
        Bs_orig.append(B_rm.clone())
        # Convert to column-major: .t().contiguous() gives (n,m) contiguous
        # cuBLAS reads it as (m,n) column-major with ldb=m
        Bs_cm.append(B_rm.t().contiguous())

    Aarray = torch.tensor([A.data_ptr() for A in As], dtype=torch.int64, device=device)
    Barray = torch.tensor([B.data_ptr() for B in Bs_cm], dtype=torch.int64, device=device)

    lda = m
    ldb = m

    result = cublasDtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount)
    assert result is not None

    # side=LEFT, uplo=UPPER, trans=N: solves upper(A)*X = alpha*B
    # A is diagonal (upper part = diagonal), so X = alpha * inv(A) * B
    for i in range(batchCount):
        inv_diag = 1.0 / torch.diag(As[i])
        # Result is in Bs_cm[i] (column-major), transpose back to row-major
        result_rm = Bs_cm[i].t().contiguous()
        expected = alpha * (inv_diag.view(-1, 1) * Bs_orig[i])
        torch.testing.assert_close(result_rm, expected, rtol=1e-5, atol=1e-5)

    print("✓ cublasDtrsmBatched test passed")