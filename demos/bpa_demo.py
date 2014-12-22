#!/usr/bin/env python

"""
Test and compare pure Python and Cython implementations of
the Bjork-Pereyra Algorithm.
"""

# Copyright (c) 2009-2014, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

from numpy import fliplr, vander, abs, arange, array
from scipy.special import chebyt

from time import time
import sys

import bionet.ted.bpa_python as bpa_python
if 'linux' in sys.platform:
    import bionet.ted.bpa_cython_linux2 as bpa_cython
elif sys.platform == 'darwin':
    import bionet.ted.bpa_cython_darwin as bpa_cython
else:
    raise RuntimeError('cannot import binary BPA module')

# Try to find the coefficients of a Chebyshev polynomial by solving
# a Vandermonde system. This test case should exhibit good accuracy for
# N less than about 13:
N = 12
i = arange(N)
a = 1.0/(i+2)
T = chebyt(N-1)
f = T(a)
V = fliplr(vander(a))
c = array(T)[::-1]

start = time()
c_solve = bpa_python.bpa(V, f)
end = time()

print 'Python implementation results:'
print 'original c = ', c
print 'solved c   = ', c_solve
print 'error      = ', abs(c-c_solve)
print 'time       = ', end-start

start = time()
c_solve = bpa_cython.bpa(V, f)
end = time()

print 'Cython implementation results:'
print 'original c = ', c
print 'solved c   = ', c_solve
print 'error      = ', abs(c-c_solve)
print 'time       = ', end-start
