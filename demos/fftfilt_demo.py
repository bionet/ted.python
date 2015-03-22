#!/usr/bin/env python

"""
Demo of fftfilt function.
"""

# Copyright (c) 2009-2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

import scipy.signal as si

import bionet.utils.signal_extras as s
import bionet.utils.band_limited as bl

print 'creating test signal..'
dur = 0.2
dt = 1e-6
fs = 1/dt
fmax = 50000.0
u = bl.gen_band_limited(dur, dt, fmax, nc=10)

print 'creating filter..'
f1 = 10000.0
a = 1
b = si.firwin(50, f1/fs)

print 'filtering with lfilter..'
u_lfilter = si.lfilter(b, a, u)

print 'filtering with fftfilt..'
u_fftfilt = s.fftfilt(b, u)

