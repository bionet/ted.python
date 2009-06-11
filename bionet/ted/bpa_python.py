#!/usr/bin/env python

"""
Python implementation of the Bjork-Pereyra algorithm for solving Vandermonde
systems.
"""

import numpy as np

def isvander(V):
    """Test if a matrix has a Vandermonde structure. 

    Parameters
    ----------
    V: numpy array
       Square matrix to be tested.

    Notes
    -----
    The matrix is assumed to be oriented such that its second column
    contains the arguments that would need to be passed to the
    vander() function in order to construct the matrix.    
    """

    (N,C) = np.shape(V)
    z = V[:,1]

    if N != C:
        raise ValueError('V must be square')

    for i in range(C):
        if any(V[:,i] != z**i):
            return False
    return True
    
def bpa(V,b):
    """Solve a Vandermonde system using the Bjork-Pereyra algorithm.

    Parameters
    ----------
    V: numpy array
        A Vandermonde matrix. 
    b: numpy array
        The system solved by this routine is dot(V,d) = b.

    Notes
    -----
    The matrix is assumed to be oriented such that its second column
    contains the arguments that would need to be passed to the
    vander() function in order to contruct the matrix.
    """
    
    (N,C) = np.shape(V)
    
    if N != C:
        raise ValueError('V must be square')
    if N <= 1:
        raise ValueError('V must contain more than 1 element')
    
    z = V[:,1].copy()
    bs = np.shape(b)
    b = b.copy().flatten()
    
    if b.size != N:
        raise ValueError('size mismatch between V and b')
    
    for n in xrange(N):
        for m in xrange(N-1,n,-1):
            b[m] = (b[m] - b[m-1])/(z[m]-z[m-n-1])
    for n in xrange(N-1,-1,-1):
        for m in xrange(n,N-1):
            b[m] = b[m] - b[m+1]*z[n]

    return np.reshape(b,bs)
