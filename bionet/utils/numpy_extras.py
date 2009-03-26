#!/usr/bin/env python

"""
Numpy Extras
============

This module contains various functions not currently included in
numpy.

- mdot            Compute the matrix product of several matricies.
- rank            Estimate the number of linearly independent rows in a matrix.
- mpower          Raise a square matrix to a (possibly non-integer) power.
- hilb            Generate a Hilbert matrix of the specified size.

"""

__all__ = ['mdot','rank','mpower','hilb']

from numpy import dot, eye, asarray, abs, shape, diag, \
     complex, float, zeros, arange, real, imag, iscomplexobj, any
from numpy.linalg import svd, eig, inv

def mdot(*args):
    '''Compute the matrix product of several multidimensional arrays
    evaluated left to right.'''

    ret = args[0]
    for a in args[1:]:
        ret = dot(ret,a)
    return ret

def rank(x,*args):
    '''Estimate the number of linearly independent rows or columns of the
    matrix x. If the tolerance is not specified, the default tolerance
    max(svd(x)[1])*max(shape(x))*1e-13 is used. Shamelessly adapted from
    the rank() function in pylab.'''

    x = asarray(x)
    s = svd(x, compute_uv=False)
    if args:
        tol = args[0]
    else:
        tol = max(abs(s))*max(shape(x))*1e-13
    return sum(s > tol)

def mpower(x,y):
    '''Compute x raised to the power y when x is a square matrix and y
    is a scalar. The matrix x must be non-defective.'''

    s = shape(x)
    if len(s) != 2 or s[0] != s[1]:
        raise ValueError('matrix must be square')
    if y == 0:
        return eye(s[0])
    [e,v] = eig(x)
    if rank(v) < s[0]:
        raise ValueError('matrix must be non-defective')

    # Need to do this because negative reals can't be raised to a
    # noninteger exponent:
    if any(e < 0):
        d = diag(asarray(e,complex)**y)
    else:
        d = diag(e**y)

    # Return a complex array only if the input array was complex or
    # the output of the computation contains complex numbers:
    result = mdot(v,d,inv(v))
    if not(iscomplexobj(x)) and not(any(imag(result))):
        return real(result)
    else:
        return result
    

def hilb(n):
    """Construct a Hilbert matrix of size n x n."""

    h = zeros((n,n),float)
    r = arange(1,n+1)
    for i in xrange(n):
        h[i,:] = 1.0/(i+r)
    return h
