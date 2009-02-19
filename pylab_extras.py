#!/usr/bin/env python

"""
Pylab Extras
============

This module contains various functions similar to those in
Matlab that are not in pylab.

"""

from numpy import finfo, single, float, double, longdouble, \
     ceil, floor, log2, abs, inf, NaN

finfo_dict = {single:finfo(single),
              float:finfo(float),
              double:finfo(double),
              longdouble:finfo(longdouble)}

def realmax(t=double):
    '''Return the largest positive floating point number representable
    with the specified precision on this computer. Double precision is
    assumed if no floating point type is specified.'''

    if t not in finfo_dict:
        raise ValueError('invalid floating point type')
    else:
        return finfo_dict[t].max

def realmin(t=double):
    '''Return the smallest positive floating point number
    representable with the specified precision on this computer.
    Double precision is assumed if no floating point type is specified.'''

    if t not in finfo_dict:
        raise ValueError('invalid floating point type')
    else:
        return finfo_dict[t].tiny.item()

def eps(x):
    '''Compute the spacing of floating point numbers.'''

    ## This function doesn't always seem to work properly. ###
    
    t = type(x)
    if t not in finfo_dict:
        raise ValueError('invalid floating point type')
    ibeta = finfo_dict[t].machar.ibeta
    maxexp = finfo_dict[t].maxexp
    machep = finfo_dict[t].machep
    minexp = finfo_dict[t].minexp
    negep = finfo_dict[t].negep
    xmax = finfo_dict[t].machar.xmax
    xmin = finfo_dict[t].machar.xmin
    
    x = abs(x)
    if x in (inf,NaN):
        return NaN
    elif x >= xmax:
        return ibeta**(maxexp+negep)
    elif x > xmin:
        return ibeta**(machep+floor(log2(x)))
    else:
        return ibeta**(minexp+negep+1)
    
