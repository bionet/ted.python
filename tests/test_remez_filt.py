#!/usr/bin/env python

"""
Filter a signal using a Remez filter.
"""

from numpy import pi
from numpy.fft import fft
from scipy.signal import lfilter, remez

import utils.signal_extras as s
import utils.gen_test_signal as g

print 'creating test signal..'
dur = 0.2
dt = 1e-6
fs = 1/dt
fmax = 5000.0
u = g.gen_test_signal(dur,dt,fmax,nc=10)
uf = fft(u)

print 'creating filter..'
f1 = 1000.0
f2 = 2000.0
a = 1
[numtaps,bands,desired,weight] = s.remezord([f1,f2],[1,0],[0.01,0.01],fs)
b = remez(numtaps,bands,desired,weight)

print 'filtering signal with lfilter..'
u_lfilter = lfilter(b,a,u)

print 'filtering signal with fftfilt..'
u_fftfilt = s.fftfilt(b,u)
