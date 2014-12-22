#!/usr/bin/env python

"""
Python implementation of the Bjork-Pereyra algorithm for solving Vandermonde
systems.
"""

# Copyright (c) 2009-2014, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

__all__ = ['bpa']

import numpy as np

def isvander(V, rtol=1e-5, atol=1e-8):
    """
    Test a matrix for Vandermonde structure.

    Test if a matrix has a Vandermonde structure by checking whether
    its columns `V[2:,:]` are integer powers of column `V[1, :]` within
    tolerance.

    Parameters
    ----------
    V : ndarray of floats, shape (M, M)
       Square matrix to be tested.
    rtol : float
       The relative tolerance parameter (see Notes).
    atol : float
       The absolute tolerance parameter (see Notes).

    Returns
    -------
    res : bool
       True if the matrix is a Vandermonde matrix, False otherwise.

    See Also
    --------
    vander, allclose

    Notes
    -----
    The matrix is assumed to be oriented such that its second column
    contains the arguments that would need to be passed to the
    `vander()` function in order to construct the matrix.

    The tolerance values are the same as those assumed by the
    `allclose()` function.
    """

    (N, C) = np.shape(V)
    z = V[:, 1]

    if N != C:
        raise ValueError('V must be square')

    for i in range(C):
        if not(np.allclose(V[:, i], z**i)):
            return False
    return True

def bpa(V, b):
    """
    Solve a Vandermonde system using BPA.

    Solve a Vandermonde linear system using the Bjork-Pereyra algorithm.

    Parameters
    ----------
    V : ndarray of floats, shape (M, M)
        A Vandermonde matrix.
    b : ndarray of floats, shape (M,)
        The system solved by this routine is `dot(V, d) == b`.

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
    `vander()` function in order to contruct the matrix.
    """

    (N, C) = np.shape(V)
    if N != C:
        raise ValueError('V must be square')
    if N <= 1:
        raise ValueError('V must contain more than 1 element')

    z = V[:, 1].copy()
    bs = np.shape(b)
    b = b.copy().flatten()

    if b.size != N:
        raise ValueError('size mismatch between V and b')

    for n in xrange(N):
        for m in xrange(N-1, n, -1):
            b[m] = (b[m] - b[m-1])/(z[m]-z[m-n-1])
    for n in xrange(N-1, -1, -1):
        for m in xrange(n, N-1):
            b[m] = b[m] - b[m+1]*z[n]

    return np.reshape(b, bs)
