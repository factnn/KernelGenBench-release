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
        _cublas_func = libcublas.cublasSdgmm
        _cublas_func.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_int,     # mode
            ctypes.c_int,     # m
            ctypes.c_int,     # n
            ctypes.c_void_p,  # A
            ctypes.c_int,     # lda
            ctypes.c_void_p,  # x
            ctypes.c_int,     # incx
            ctypes.c_void_p,  # C
            ctypes.c_int      # ldc
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSdgmm(mode, m, n, A, lda, x, incx, C, ldc):
    '''ctypes cuBLAS C API baseline for cublasSdgmm'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Call cuBLAS C API
    func(handle, mode, m, n, A_ptr, lda, x_ptr, incx, C_ptr, ldc)

    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    m_orig, n_orig = 4, 6
    A = torch.randn(m_orig, n_orig, dtype=torch.float32, device='cuda').contiguous()
    A_clone = A.clone()

    CUBLAS_SIDE_LEFT = 0
    CUBLAS_SIDE_RIGHT = 1

    # Column-major trick:
    # Row-major A(m,n) is read as column-major (n,m) by cuBLAS.
    # cuBLAS m_param=n_orig, n_param=m_orig, lda=n_orig, ldc=n_orig

    # --- Right multiplication in row-major: C = A * diag(x_right) ---
    # x_right has n_orig elements (one per column)
    # In column-major view (n,m): RIGHT means scale each column of A_cm by x[col_idx]
    # But columns of A_cm correspond to rows of A_rm.
    # Actually: cuBLAS LEFT on A_cm(n,m) with x(n) scales each row of A_cm => each column of A_rm ✓
    x_right = torch.randn(n_orig, dtype=torch.float32, device='cuda')
    x_right_clone = x_right.clone()
    C_right = torch.empty_like(A)

    out_right = cublasSdgmm(CUBLAS_SIDE_LEFT, n_orig, m_orig, A, n_orig, x_right, 1, C_right, n_orig)
    assert out_right is not None
    expected_right = A_clone * x_right_clone.view(1, -1)
    torch.testing.assert_close(out_right, expected_right, rtol=1e-5, atol=1e-5)

    # --- Left multiplication in row-major: C = diag(x_left) * A ---
    # x_left has m_orig elements (one per row)
    # cuBLAS RIGHT on A_cm(n,m) with x(m): C_cm[i,j] = A_cm[i,j]*x[j]
    # In row-major: C_rm[j,i] = A_rm[j,i]*x[j] => row j of A_rm scaled by x[j] ✓
    x_left = torch.randn(m_orig, dtype=torch.float32, device='cuda')
    x_left_clone = x_left.clone()
    C_left = torch.empty_like(A)

    out_left = cublasSdgmm(CUBLAS_SIDE_RIGHT, n_orig, m_orig, A, n_orig, x_left, 1, C_left, n_orig)
    assert out_left is not None
    expected_left = x_left_clone.view(-1, 1) * A_clone
    torch.testing.assert_close(out_left, expected_left, rtol=1e-5, atol=1e-5)

    print("✓ cublasSdgmm test passed")