#!/usr/bin/env python

"""
Scipy Extras
============

This module contains various functions not currently included in
scipy.

- ei              Compute the exponential integral of a complex value.
- si              Compute the sine integral of a complex value.
- ci              Compute the cosine integral of a complex value.
- shi             Compute the hyperbolic sine of a complex value.
- chi             Compute the hyperbolic cosine of a complex value.
"""

__all__ = ['ei','si','ci','li','shi','chi']

from numpy import pi, log, asarray, complex, iscomplexobj, real, \
     iterable, asscalar, any
from scipy.special import exp1

def ei(z):
    """Compute the exponential integral of a complex value."""

    # XXX: should return 0 for -inf, inf for +inf, and -inf for 0
    zc = asarray(z,complex)
    res = -exp1(-zc)+(log(zc)-log(1.0/zc))/2.0-log(-zc)
    if not iscomplexobj(z):
        res = real(res)
    if iterable(z):
        return res
    else:
        return asscalar(res)

def si(z):
    """Compute the sine integral of a complex value."""

    # XXX: should return 0 for 0, pi/2 for +inf, -pi/2 for -inf
    zc = asarray(z,complex)
    res = (1j/2)*(exp1(-1j*zc)-exp1(1j*zc)+log(-1j*zc)-log(1j*zc))
    if not iscomplexobj(z):
        res = real(res)
    if iterable(z):
        return res
    else:
        return asscalar(res)

def ci(z):
    """Compute the cosine integral of a complex value."""

    # XXX: should return -inf for 0, 0 for +inf, and pi*1j for -inf
    zc = asarray(z,complex)
    res = log(zc)-(exp1(-1j*zc)+exp1(1j*zc)+log(-1j*zc)+log(1j*zc))/2.0
    if not iscomplexobj(z):
        res = real(res)
    if iterable(z):
        return res
    else:
        return asscalar(res)

def li(z):
    """Compute the logarithmic integral of a complex value."""

    # XXX: should return -inf for 1 and +inf for +inf
    zc = asarray(z,complex)
    res = -exp1(-log(zc))+(log(log(zc))-log(1/log(zc)))/2.0-log(-log(zc))
    if not iscomplexobj(z) and not any(z < 0):
        res = real(res)
    if iterable(z):
        return res
    else:
        return asscalar(res)

def shi(z):
    """Compute the hyperbolic sine integral of a complex value."""

    # XXX: should return 0 for 0, +inf for +inf, -inf for -inf
    zc = asarray(z,complex)
    res = (exp1(zc)-exp1(-zc)-log(-zc)+log(zc))/2.0
    if not iscomplexobj(z):
        res = real(res)
    if iterable(z):
        return res
    else:
        return asscalar(res)

def chi(z):
    """Compute the hyperbolic cosine integral of a complex value."""

    # XXX: should return -inf for 0, +inf for +inf, +inf for -inf
    zc = asarray(z,complex)
    res = -(exp1(-zc)+exp1(zc)+log(-zc)-log(zc))/2.0
    if not iscomplexobj(z):
        res = real(res)
    if iterable(z):
        return res
    else:
        return asscalar(res)
