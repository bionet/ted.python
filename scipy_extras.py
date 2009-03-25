#!/usr/bin/env python

"""
Scipy Extras
============

This module contains various functions not currently included in
scipy.

- ei              Compute the exponential integral of a complex value.

"""

__all__ = ['ei']

from numpy import pi,log,asarray,complex,iscomplexobj,real,iterable,asscalar
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
