#!/usr/bin/env python

"""
Demo of fftfilt function.
"""

import scipy.signal as si

import bionet.utils.signal_extras as s
import bionet.utils.gen_test_signal as g

print 'creating test signal..'
dur = 0.2
dt = 1e-6
fs = 1/dt
fmax = 50000.0
u = g.gen_test_signal(dur,dt,fmax,nc=10)

print 'creating filter..'
f1 = 10000.0
a = 1
b = si.firwin(50,f1/fs)

print 'filtering with lfilter..'
u_lfilter = si.lfilter(b,a,u)

print 'filtering with fftfilt..'
u_fftfilt = s.fftfilt(b,u)

