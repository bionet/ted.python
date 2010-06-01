# -*- Mode: python -*- 

__all__ = ['bpa']

include "numpy.pxd"
import numpy as np

def bpa(ndarray V, ndarray b):
    """
    bpa(V, b)
    
    Solve a Vandermonde system using BPA.
    
    Solve a Vandermonde system using the Bjork-Pereyra algorithm.

    Parameters
    ----------
    V : ndarray of floats, shape (M, M)
        A Vandermonde matrix. 
    b : ndarray of floats, shape (M,)
        The system solved by this routine is `dot(V,d) == b`.

    Returns
    -------
    d : ndarray of floats, shape (M,)
        System solution.
        
    See Also
    --------
    numpy.linalg.solve
    
    Notes
    -----
    The matrix is assumed to be oriented such that its second column
    contains the arguments that would need to be passed to the
    `vander()` function in order to construct the matrix.
    
    """

    cdef int N, C
    N = V.shape[0]
    C = V.shape[1]
    if N <> C:
        raise ValueError('V must be square')
    if N <= 1:
        raise ValueError('V must contain more than 1 element')

    # Save shape of b to set the shape of the output:
    bs = np.shape(b)

    # Copy the input values to avoid modifying them:
    cdef ndarray z_array
    cdef ndarray b_array
    z_array = np.array(V[:, 1], np.complex)
    b_array = np.array(b.flatten(), np.complex)

    if b_array.shape[0] <> N:
        raise ValueError('size mismatch between V and b')

    cdef double complex *z_data
    cdef double complex *b_data
    z_data = <double complex *>z_array.data
    b_data = <double complex *>b_array.data

    cdef int n, m
    for n from 0 <= n < N:
        for m from N-1 >= m > n:
            b_data[m] = (b_data[m]-b_data[m-1])/(z_data[m]-z_data[m-n-1])
    for n from N-1 >= n > -1:
        for m from n <= m < N-1:
            b_data[m] = b_data[m]-(b_data[m+1]*z_data[n])

    if np.iscomplexobj(V) or np.iscomplexobj(b):
        return np.reshape(b_array, bs)
    else:
        return np.reshape(np.real(b_array), bs)

