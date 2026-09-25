import torch
import ctypes
import os

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_set_pointer_mode = None
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

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
        _cublas_func = libcublas.cublasCgemmStridedBatched_64
        _cublas_func.argtypes = [
            ctypes.c_void_p,           # handle
            ctypes.c_int,              # transa
            ctypes.c_int,              # transb
            ctypes.c_longlong,         # m
            ctypes.c_longlong,         # n
            ctypes.c_longlong,         # k
            ctypes.POINTER(cuComplex), # alpha
            ctypes.POINTER(cuComplex), # A
            ctypes.c_longlong,         # lda
            ctypes.c_longlong,         # strideA
            ctypes.POINTER(cuComplex), # B
            ctypes.c_longlong,         # ldb
            ctypes.c_longlong,         # strideB
            ctypes.POINTER(cuComplex), # beta
            ctypes.POINTER(cuComplex), # C
            ctypes.c_longlong,         # ldc
            ctypes.c_longlong,         # strideC
            ctypes.c_longlong          # batchCount
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, complex(value)) if getattr(dtype, "is_complex", False) else (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCgemmStridedBatched_64(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    '''ctypes cuBLAS C API baseline for cublasCgemmStridedBatched_64'''
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Map transposition inputs if provided as strings
    if isinstance(transa, str):
        transa = 0 if transa == 'N' else (1 if transa == 'T' else 2)
    if isinstance(transb, str):
        transb = 0 if transb == 'N' else (1 if transb == 'T' else 2)

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors for alpha and beta
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa),
        ctypes.c_int(transb),
        ctypes.c_longlong(m),
        ctypes.c_longlong(n),
        ctypes.c_longlong(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuComplex)),
        ctypes.cast(A_ptr, ctypes.POINTER(cuComplex)),
        ctypes.c_longlong(lda),
        ctypes.c_longlong(strideA),
        ctypes.cast(B_ptr, ctypes.POINTER(cuComplex)),
        ctypes.c_longlong(ldb),
        ctypes.c_longlong(strideB),
        ctypes.cast(beta_ptr, ctypes.POINTER(cuComplex)),
        ctypes.cast(C_ptr, ctypes.POINTER(cuComplex)),
        ctypes.c_longlong(ldc),
        ctypes.c_longlong(strideC),
        ctypes.c_longlong(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasCgemmStridedBatched_64 failed with status {status}")
    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    batch = 4
    m, n, k = 5, 6, 3

    # Create test tensors on GPU with complex64 dtype (row-major)
    A_rm = torch.randn(batch, m, k, device='cuda', dtype=torch.float32) + 1j * torch.randn(batch, m, k, device='cuda', dtype=torch.float32)
    B_rm = torch.randn(batch, k, n, device='cuda', dtype=torch.float32) + 1j * torch.randn(batch, k, n, device='cuda', dtype=torch.float32)
    C_rm = torch.randn(batch, m, n, device='cuda', dtype=torch.float32) + 1j * torch.randn(batch, m, n, device='cuda', dtype=torch.float32)

    # Clone originals for comparison
    A_rm_orig = A_rm.clone()
    B_rm_orig = B_rm.clone()
    C_rm_orig = C_rm.clone()

    # Prepare column-major equivalents by transposing (so row-major of transposed acts as column-major)
    A_cm = A_rm.transpose(-1, -2).contiguous()  # shape (batch, k, m) -> column-major (m, k)
    B_cm = B_rm.transpose(-1, -2).contiguous()  # shape (batch, n, k) -> column-major (k, n)
    C_cm = C_rm.transpose(-1, -2).contiguous()  # shape (batch, n, m) -> column-major (m, n)

    # Set leading dimensions and strides for column-major
    lda = m
    ldb = k
    ldc = m
    strideA = lda * k
    strideB = ldb * n
    strideC = ldc * n
    batchCount = batch

    # Scalars
    alpha = complex(0.7, 0.2)
    beta = complex(-0.3, 0.4)

    # Call baseline
    C_out_cm = cublasCgemmStridedBatched_64('N', 'N', m, n, k, alpha, A_cm, lda, strideA, B_cm, ldb, strideB, beta, C_cm, ldc, strideC, batchCount)

    assert C_out_cm is not None

    # PyTorch reference in row-major
    expected_rm = alpha * torch.matmul(A_rm_orig, B_rm_orig) + beta * C_rm_orig

    # Convert output back to row-major layout
    C_out_rm = C_out_cm.transpose(-1, -2).contiguous()

    # Numerical check
    torch.testing.assert_close(C_out_rm, expected_rm, rtol=1e-5, atol=1e-5)
    print("✓ cublasCgemmStridedBatched_64 test passed")