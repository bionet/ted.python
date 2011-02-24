#!/usr/bin/env python

"""
Function for computing the product of the (Hermitian) transpose of a
matrix and itself in PyCUDA without requiring the explicit creation of
the transposed matrix.
"""

import pycuda.gpuarray as gpuarray
import numpy as np

import scikits.cuda.cuda as cuda
import scikits.cuda.cublas as cublas

def prodtrans(a_gpu):
    """
    Compute the product of the (Hermitian) transpose of a matrix and itself.

    Parameters
    ----------
    a_gpu : pycuda.gpuarray.GPUArray
        Input array.

    Returns
    -------
    c_gpu : pycuda.gpuarray.GPUArray float{32,64}, or complex{64,128}
        Dot product of `a_gpu` and `b_gpu`. When the inputs are 1D
        arrays, the result will be returned as a scalar.

    Examples
    --------
    >>> import pycuda.gpuarray as gpuarray
    >>> import pycuda.autoinit
    >>> import numpy as np
    >>> import scikits.cuda.cublas as cublas
    >>> cublas.cublasInit()
    >>> a = np.asarray(np.random.rand(4, 2), np.float32)
    >>> a_gpu = gpuarray.to_gpu(a)
    >>> c_gpu = prodtrans(a_gpu)
    >>> np.allclose(np.dot(a.T, a), c_gpu.get())
    True

    """

    if len(a_gpu.shape) == 1:

        # Compute inner product for 1D arrays:
        if a_gpu.dtype == np.complex64:
            cublas_func = cublas._libcublas.cublasCdotu
        elif a_gpu.dtype == np.float32:
            cublas_func = cublas._libcublas.cublasSdot
        elif a_gpu.dtype == np.complex128:
            cublas_func = cublas._libcublas.cublasZdotu
        elif a_gpu.dtype == np.float64:
            cublas_func = cublas._libcublas.cublasDdot
        else:
            raise ValueError('unrecognized input type')

        result = cublas_func(a_gpu.size, int(a_gpu.gpudata), 1,
                             int(a_gpu.gpudata), 1)

        if a_gpu.dtype == np.complex64:
            return np.float32(result.x)+1j*np.float32(result.y)
        elif a_gpu.dtype == np.complex128:
            return np.float64(result.x)+1j*np.float64(result.y)
        elif a_gpu.dtype == np.float32:
            return np.float32(result)
        else:
            return np.float64(result)
    else:

        # Perform matrix multiplication for 2D arrays:
        if a_gpu.dtype == np.complex64:
            cublas_func = cublas._libcublas.cublasCgemm        
            alpha = cuda.cuFloatComplex(1, 0)
            beta = cuda.cuFloatComplex(0, 0)
        elif a_gpu.dtype == np.float32: 
            cublas_func = cublas._libcublas.cublasSgemm
            alpha = np.float32(1.0)
            beta = np.float32(0.0)
        elif a_gpu.dtype == np.complex128:
            cublas_func = cublas._libcublas.cublasZgemm        
            alpha = cuda.cuDoubleComplex(1, 0)
            beta = cuda.cuDoubleComplex(0, 0)
        elif a_gpu.dtype == np.float64:
            cublas_func = cublas._libcublas.cublasDgemm
            alpha = np.float64(1.0)
            beta = np.float64(0.0)
        else:
            raise ValueError('unrecognized input type')

        transa = 'N'
        transb = 'C'
        m = a_gpu.shape[1]
        n = a_gpu.shape[1]
        k = a_gpu.shape[0]
        lda = max(1, m)
        ldb = max(1, n)
        ldc = max(1, m)

        c_gpu = gpuarray.zeros((a_gpu.shape[1], a_gpu.shape[1]), a_gpu.dtype)
        cublas_func(transa, transb, m, n, k, alpha, int(a_gpu.gpudata),
                    lda, int(a_gpu.gpudata), ldb, beta, int(c_gpu.gpudata), ldc)

        status = cublas.cublasGetError()
        cublas.cublasCheckStatus(status)

        return c_gpu

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
