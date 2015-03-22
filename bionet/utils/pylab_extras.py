#!/usr/bin/env python

"""
Pylab Extras
============
This module contains various functions similar to those in
Matlab that are not in pylab.

- eps             Compute spacing of floating point numbers.
- minmax          Return range of array.
- realmax         Return largest representable positive floating point number.
- realmin         Return smallest representable positive floating point number.
"""

# Copyright (c) 2009-2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

__all__ = ['eps', 'realmax', 'realmin']

from numpy import array, finfo, single, float, double, longdouble, \
     floor, log2, abs, inf, NaN, min, max, shape, vstack

finfo_dict = {single:finfo(single),
              float:finfo(float),
              double:finfo(double),
              longdouble:finfo(longdouble)}

def realmax(t=double):
    """Return the largest positive floating point number representable
    with the specified precision on this computer. Double precision is
    assumed if no floating point type is specified."""

    if t not in finfo_dict:
        raise ValueError('invalid floating point type')
    else:
        return finfo_dict[t].max

def realmin(t=double):
    """Return the smallest positive floating point number
    representable with the specified precision on this computer.
    Double precision is assumed if no floating point type is specified."""

    if t not in finfo_dict:
        raise ValueError('invalid floating point type')
    else:
        return finfo_dict[t].tiny.item()

def eps(x):
    """Compute the spacing of floating point numbers."""
    
    t = type(x)
    if t not in finfo_dict:
        raise ValueError('invalid floating point type')

    ibeta = int(finfo_dict[t].machar.ibeta)
    maxexp = finfo_dict[t].maxexp
    machep = finfo_dict[t].machep
    minexp = finfo_dict[t].minexp
    negep = finfo_dict[t].negep
    xmax = finfo_dict[t].machar.xmax
    xmin = finfo_dict[t].machar.xmin
    
    x = abs(x)
    if x in (inf, NaN):
        return NaN
    elif x >= xmax:
        return ibeta**(maxexp+negep)
    elif x > xmin:

	# Convert output of log2() to int to prevent 
	# imprecision from confusing floor():
        return ibeta**(machep+int(floor(int(log2(x)))))
    else:
        return ibeta**(minexp+negep+1)
    
def minmax(x):
    """Return the range of the given array. If the array has 2
    dimensions, return an array containing the minima and maxima of
    each of the rows."""

    dims = len(shape(x))
    if dims == 1:
        return array((min(x), max(x)))
    elif dims == 2:
        return vstack((min(x,1), max(x,1))).T
    else:
        raise ValueError('undefined for arrays with more than 2 dimensions')
