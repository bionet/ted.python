#!/usr/bin/env python

"""
Numpy Extras
============
This module contains various functions not currently included in
numpy [1]_.

- crand           Generate complex uniformly distributed random values.
- hilb            Generate a Hilbert matrix of the specified size.
- iceil           Ceiling of the input returned as an integer.
- ifloor          Floor of the input returned as an integer.
- iround          Input rounded to the nearest integer
- mdot            Compute the matrix product of several matricies.
- mpower          Raise a square matrix to a (possibly non-integer) power.
- rank            Estimate the number of linearly independent rows in a matrix.

.. [1] http://numpy.scipy.org/
"""

# Copyright (c) 2009-2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

__all__ = ['crand', 'iceil', 'ifloor', 'iround', 'mdot', 'rank', 'mpower', 'hilb']

import numpy as np

def crand(*args):
    """
    Complex random values in a given shape.

    Create an array of the given shape whose entries are complex
    numbers with real and imaginary parts sampled from a uniform
    distribution over ``[0, 1)``.

    Parameters
    ----------
    d0, d1, ..., dn : int
        Shape of the output.

    Returns
    -------
    out : numpy.ndarray
        Complex random variables.
    """

    return np.random.rand(*args)+1j*np.random.rand(*args)

def iceil(x):
    """
    Return the ceiling of the input, element-wise.

    The ceil of the scalar `x` is the smallest integer `i`, such
    that `i >= x`.  It is often denoted as :math:`\lceil x \rceil`.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    y : {numpy.ndarray, scalar}
        The ceiling of each element in `x`, with `int` dtype.
    """

    return np.ceil(x).astype(int)

def ifloor(x):
    """
    Return the floor of the input, element-wise.

    The floor of the scalar `x` is the largest integer `i`, such
    that `i <= x`.  It is often denoted as :math:`\lfloor x \rfloor`.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    y : {numpy.ndarray, scalar}
        The floor of each element in `x`, with `int` dtype.
    """

    return np.floor(x).astype(int)

def iround(x):
    """
    Round an array to the nearest integer.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    y : {numpy.ndarray, scalar}
        The rounded elements in `x`, with `int` dtype.
    """

    return np.round(x).astype(int)

def mdot(*args):
    """
    Dot product of several arrays.

    Compute the dot product of several arrays in the order they are
    listed.
    """

    ret = args[0]
    for a in args[1:]:
        ret = np.dot(ret, a)
    return ret

def rank(x, *args):
    """
    Compute matrix rank.

    Estimate the number of linearly independent rows or columns of the
    matrix x.

    Parameters
    ----------
    x : array_like, shape `(M, N)`
        Matrix to analyze.
    tol : float
        Tolerance; the default is `max(svd(x)[1])*max(shape(x))*1e-13`

    Returns
    -------
    r : int
        Estimated rank of matrix.
    """

    x = np.asarray(x)
    s = np.linalg.svd(x, compute_uv=False)
    if args:
        tol = args[0]
    else:
        tol = np.max(np.abs(s))*np.max(np.shape(x))*1e-13
    return sum(s > tol)

def mpower(x, y):
    """
    Matrix power function.

    Compute `x` raised to the power `y` where `x` is a square matrix and `y`
    is a scalar.

    Notes
    -----
    The matrix `x` must be non-defective.

    """

    s = np.shape(x)
    if len(s) != 2 or s[0] != s[1]:
        raise ValueError('matrix must be square')
    if y == 0:
        return np.eye(s[0])
    [e, v] = np.linalg.eig(x)
    if rank(v) < s[0]:
        raise ValueError('matrix must be non-defective')

    # Need to do this because negative reals can't be raised to a
    # noninteger exponent:
    if np.any(e < 0):
        d = np.diag(np.asarray(e, np.complex)**y)
    else:
        d = np.diag(e**y)

    # Return a complex array only if the input array was complex or
    # the output of the computation contains complex numbers:
    result = mdot(v, d, np.linalg.inv(v))
    if not(np.iscomplexobj(x)) and not(np.any(np.imag(result))):
        return np.real(result)
    else:
        return result

def hilb(n):
    """
    Construct a Hilbert matrix.

    Parameters
    ----------
    n : int
        Number of rows and columns in matrix.

    Returns
    -------
    h : numpy.ndarray
        Generated Hilbert matrix of shape `(n, n)`.
    """

    h = np.empty((n, n), float)
    r = np.arange(1, n+1)
    for i in xrange(n):
        h[i, :] = 1.0/(i+r)
    return h

