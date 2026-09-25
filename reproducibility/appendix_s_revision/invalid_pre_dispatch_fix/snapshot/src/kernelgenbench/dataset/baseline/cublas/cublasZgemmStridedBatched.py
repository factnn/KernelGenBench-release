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

# cuBLAS operation enums
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

# cuDoubleComplex struct for ctypes casting
class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

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
        func = libcublas.cublasZgemmStridedBatched
        func.argtypes = [
            ctypes.c_void_p,                # handle
            ctypes.c_int,                   # transa
            ctypes.c_int,                   # transb
            ctypes.c_int,                   # m
            ctypes.c_int,                   # n
            ctypes.c_int,                   # k
            ctypes.POINTER(cuDoubleComplex),# alpha
            ctypes.POINTER(cuDoubleComplex),# A
            ctypes.c_int,                   # lda
            ctypes.c_longlong,              # strideA
            ctypes.POINTER(cuDoubleComplex),# B
            ctypes.c_int,                   # ldb
            ctypes.c_longlong,              # strideB
            ctypes.POINTER(cuDoubleComplex),# beta
            ctypes.POINTER(cuDoubleComplex),# C
            ctypes.c_int,                   # ldc
            ctypes.c_longlong,              # strideC
            ctypes.c_int                    # batchCount
        ]
        func.restype = ctypes.c_int
        _cublas_func = func
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, complex(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasZgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    '''ctypes cuBLAS C API baseline for cublasZgemmStridedBatched'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert string trans to int if needed (N->0, T->1)
    if isinstance(transa, str):
        transa = 0 if transa == 'N' else 1
    if isinstance(transb, str):
        transb = 0 if transb == 'N' else 1

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors for alpha and beta
    alpha_gpu = _get_scalar_gpu('alpha', complex(alpha), torch.complex128)
    beta_gpu = _get_scalar_gpu('beta', complex(beta), torch.complex128)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa),
        ctypes.c_int(transb),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(A_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.c_int(lda),
        ctypes.c_longlong(strideA),
        ctypes.cast(B_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.c_int(ldb),
        ctypes.c_longlong(strideB),
        ctypes.cast(beta_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(C_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.c_int(ldc),
        ctypes.c_longlong(strideC),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasZgemmStridedBatched failed with status {status}")

    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required for this test."

    # Dimensions
    batchCount = 4
    m, n, k = 5, 6, 7
    alpha = 1.25 + 0.5j
    beta = -0.75 + 0.2j

    # Create batched complex128 tensors in row-major (PyTorch default)
    A = (torch.randn(batchCount, m, k, dtype=torch.float64, device='cuda') +
         1j * torch.randn(batchCount, m, k, dtype=torch.float64, device='cuda')).to(torch.complex128)
    B = (torch.randn(batchCount, k, n, dtype=torch.float64, device='cuda') +
         1j * torch.randn(batchCount, k, n, dtype=torch.float64, device='cuda')).to(torch.complex128)
    C = (torch.randn(batchCount, m, n, dtype=torch.float64, device='cuda') +
         1j * torch.randn(batchCount, m, n, dtype=torch.float64, device='cuda')).to(torch.complex128)

    # Clone originals for reference
    A_orig = A.clone()
    B_orig = B.clone()
    C_orig = C.clone()

    # Convert to column-major by transposing per batch and making contiguous
    # Row-major (m, k) -> Column-major (m, k) represented as row-major (k, m)
    A_cm = A.transpose(-2, -1).contiguous()  # shape (batch, k, m)
    B_cm = B.transpose(-2, -1).contiguous()  # shape (batch, n, k)
    C_cm = C.transpose(-2, -1).contiguous()  # shape (batch, n, m)

    # Leading dimensions for column-major storage
    lda = m  # rows of A
    ldb = k  # rows of B
    ldc = m  # rows of C

    # Operations: No transpose
    transa = CUBLAS_OP_N
    transb = CUBLAS_OP_N

    # Strides in number of elements (not bytes), for column-major storage
    # For cuBLAS GEMM: storage dims depend on transa/transb
    strideA = lda * (k if transa == CUBLAS_OP_N else m)
    strideB = ldb * (n if transb == CUBLAS_OP_N else k)
    strideC = ldc * n

    # Invoke baseline
    result_cm = cublasZgemmStridedBatched(
        transa, transb,
        m, n, k,
        alpha,
        A_cm, lda, strideA,
        B_cm, ldb, strideB,
        beta,
        C_cm, ldc, strideC,
        batchCount
    )

    assert result_cm is not None

    # Convert result back to row-major for comparison
    result = result_cm.transpose(-2, -1).contiguous()

    # PyTorch reference in row-major
    expected = alpha * torch.matmul(A_orig, B_orig) + beta * C_orig

    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    print("✓ cublasZgemmStridedBatched test passed")