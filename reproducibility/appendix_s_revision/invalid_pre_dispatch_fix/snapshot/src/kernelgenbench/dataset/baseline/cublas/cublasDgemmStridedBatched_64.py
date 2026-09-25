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
        _cublas_func = libcublas.cublasDgemmStridedBatched_64
        _cublas_func.argtypes = [
            ctypes.c_void_p,                # handle
            ctypes.c_int,                   # transa
            ctypes.c_int,                   # transb
            ctypes.c_longlong,              # m
            ctypes.c_longlong,              # n
            ctypes.c_longlong,              # k
            ctypes.POINTER(ctypes.c_double),# alpha
            ctypes.POINTER(ctypes.c_double),# A
            ctypes.c_longlong,              # lda
            ctypes.c_longlong,              # strideA
            ctypes.POINTER(ctypes.c_double),# B
            ctypes.c_longlong,              # ldb
            ctypes.c_longlong,              # strideB
            ctypes.POINTER(ctypes.c_double),# beta
            ctypes.POINTER(ctypes.c_double),# C
            ctypes.c_longlong,              # ldc
            ctypes.c_longlong,              # strideC
            ctypes.c_longlong               # batchCount
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDgemmStridedBatched_64(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    '''ctypes cuBLAS C API baseline for cublasDgemmStridedBatched_64'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Map string operations to cublasOperation_t integers if needed
    if isinstance(transa, str):
        transa = 0 if transa == 'N' else (1 if transa == 'T' else 2)
    if isinstance(transb, str):
        transb = 0 if transb == 'N' else (1 if transb == 'T' else 2)

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa),
        ctypes.c_int(transb),
        ctypes.c_longlong(m),
        ctypes.c_longlong(n),
        ctypes.c_longlong(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_longlong(lda),
        ctypes.c_longlong(strideA),
        ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_longlong(ldb),
        ctypes.c_longlong(strideB),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_longlong(ldc),
        ctypes.c_longlong(strideC),
        ctypes.c_longlong(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasDgemmStridedBatched_64 failed with status {status}")
    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'
    dtype = torch.float64

    m, n, k = 4, 5, 3
    batchCount = 2

    A_rm = torch.randn(batchCount, m, k, device=device, dtype=dtype).contiguous()
    B_rm = torch.randn(batchCount, k, n, device=device, dtype=dtype).contiguous()
    C_rm = torch.randn(batchCount, m, n, device=device, dtype=dtype).contiguous()

    A_ref = A_rm.clone()
    B_ref = B_rm.clone()
    C_ref = C_rm.clone()

    alpha = 1.3
    beta = 0.7

    # Column-major trick: row-major C(m,n) = alpha*A(m,k)@B(k,n) + beta*C
    # => cuBLAS: C^T(n,m) = alpha*B^T(n,k)@A^T(k,m) + beta*C^T(n,m)
    # Pass row-major tensors directly, swap A<->B, m'=n, n'=m
    # B_rm(batch,k,n) as A operand: each slice is (k,n), cuBLAS reads as column-major (n,k), lda=n
    # A_rm(batch,m,k) as B operand: each slice is (m,k), cuBLAS reads as column-major (k,m), ldb=k
    # C_rm(batch,m,n) as C: each slice is (m,n), cuBLAS reads as column-major (n,m), ldc=n
    lda = n
    ldb = k
    ldc = n
    strideA = k * n
    strideB = m * k
    strideC = m * n

    C_out = cublasDgemmStridedBatched_64(
        'N', 'N', n, m, k, alpha,
        B_rm, lda, strideA,
        A_rm, ldb, strideB,
        beta,
        C_rm, ldc, strideC,
        batchCount
    )
    assert C_out is not None

    expected = alpha * torch.matmul(A_ref, B_ref) + beta * C_ref
    torch.testing.assert_close(C_rm, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDgemmStridedBatched_64 test passed")