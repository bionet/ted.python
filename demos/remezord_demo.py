#!/usr/bin/env python

"""
Remez filter construction demo.
"""

from numpy.fft import fft
from scipy.signal import lfilter, remez

import bionet.utils.signal_extras as s
import bionet.utils.band_limited as bl

print 'creating test signal..'
dur = 0.2
dt = 1e-6
fs = 1/dt
fmax = 5000.0
u = bl.gen_band_limited(dur, dt, fmax, nc=10)
uf = fft(u)

print 'creating filter..'
f1 = 1000.0
f2 = 2000.0
a = 1
[numtaps, bands, desired, weight] = s.remezord([f1, f2], [1, 0],
                                               [0.01, 0.01], fs)
b = remez(numtaps, bands, desired, weight)

print 'filtering signal with lfilter..'
u_lfilter = lfilter(b, a, u)

print 'filtering signal with fftfilt..'
u_fftfilt = s.fftfilt(b, u)
