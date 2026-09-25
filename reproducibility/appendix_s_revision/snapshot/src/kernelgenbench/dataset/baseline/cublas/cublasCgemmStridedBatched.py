import torch
import ctypes
import os

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

# Define cuComplex type for ctypes (float2)
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

def _get_cublas_func():
    '''Get cuBLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasCgemmStridedBatched
        _cublas_func.argtypes = [
            ctypes.c_void_p,              # handle
            ctypes.c_int,                 # transa
            ctypes.c_int,                 # transb
            ctypes.c_int,                 # m
            ctypes.c_int,                 # n
            ctypes.c_int,                 # k
            ctypes.POINTER(cuComplex),    # alpha
            ctypes.POINTER(cuComplex),    # A
            ctypes.c_int,                 # lda
            ctypes.c_longlong,            # strideA
            ctypes.POINTER(cuComplex),    # B
            ctypes.c_int,                 # ldb
            ctypes.c_longlong,            # strideB
            ctypes.POINTER(cuComplex),    # beta
            ctypes.POINTER(cuComplex),    # C
            ctypes.c_int,                 # ldc
            ctypes.c_longlong,            # strideC
            ctypes.c_int                  # batchCount
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include the scalar value so different alpha/beta values get different tensors
    if dtype in (torch.complex64, torch.complex128):
        cache_key = (key, dtype, complex(value))
    else:
        cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    '''ctypes cuBLAS C API baseline for cublasCgemmStridedBatched'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Map transposition flags if provided as strings
    if isinstance(transa, str):
        transa = {'N': 0, 'T': 1, 'C': 2}[transa]
    if isinstance(transb, str):
        transb = {'N': 0, 'T': 1, 'C': 2}[transb]

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors for alpha and beta (complex64)
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa),
        ctypes.c_int(transb),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuComplex)),
        ctypes.cast(A_ptr, ctypes.POINTER(cuComplex)),
        ctypes.c_int(lda),
        ctypes.c_longlong(strideA),
        ctypes.cast(B_ptr, ctypes.POINTER(cuComplex)),
        ctypes.c_int(ldb),
        ctypes.c_longlong(strideB),
        ctypes.cast(beta_ptr, ctypes.POINTER(cuComplex)),
        ctypes.cast(C_ptr, ctypes.POINTER(cuComplex)),
        ctypes.c_int(ldc),
        ctypes.c_longlong(strideC),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasCgemmStridedBatched failed with status {status}")
    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    batchCount = 4
    m, n, k = 5, 6, 3
    alpha = 1.5 + 0.25j
    beta = -0.3 + 0.1j

    # Create test data in row-major (PyTorch default) on GPU
    A_row = (torch.randn(batchCount, m, k, device='cuda', dtype=torch.float32) +
             1j * torch.randn(batchCount, m, k, device='cuda', dtype=torch.float32)).to(torch.complex64)
    B_row = (torch.randn(batchCount, k, n, device='cuda', dtype=torch.float32) +
             1j * torch.randn(batchCount, k, n, device='cuda', dtype=torch.float32)).to(torch.complex64)
    C_row = (torch.randn(batchCount, m, n, device='cuda', dtype=torch.float32) +
             1j * torch.randn(batchCount, m, n, device='cuda', dtype=torch.float32)).to(torch.complex64)

    # Clone originals for reference
    A_row_ref = A_row.clone()
    B_row_ref = B_row.clone()
    C_row_ref = C_row.clone()

    # Prepare column-major equivalent buffers via transposition (row-major (k,m) == col-major (m,k))
    At = A_row.transpose(1, 2).contiguous()  # shape (batch, k, m)
    Bt = B_row.transpose(1, 2).contiguous()  # shape (batch, n, k)
    Ct = C_row.transpose(1, 2).contiguous()  # shape (batch, n, m)

    # Leading dimensions and strides for column-major interpretation
    lda = m
    ldb = k
    ldc = m
    strideA = lda * k
    strideB = ldb * n
    strideC = ldc * n

    # Invoke baseline function (column-major, no transpose)
    Ct_out = cublasCgemmStridedBatched(
        'N', 'N',
        m, n, k,
        alpha,
        At, lda, strideA,
        Bt, ldb, strideB,
        beta,
        Ct, ldc, strideC,
        batchCount
    )

    assert Ct_out is not None

    # Convert result back to row-major layout for comparison
    C_row_out = Ct_out.transpose(1, 2).contiguous()

    # PyTorch reference in row-major
    expected = alpha * torch.matmul(A_row_ref, B_row_ref) + beta * C_row_ref

    torch.testing.assert_close(C_row_out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCgemmStridedBatched test passed")