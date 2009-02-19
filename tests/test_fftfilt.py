#!/usr/bin/env python

"""
Test fftfilt function.
"""

from numpy import pi
from scipy.signal import lfilter, remez, firwin

import utils.signal_extras as s
import utils.gen_test_signal as g

print 'creating test signal..'
dur = 0.2
dt = 1e-6
fs = 1/dt
fmax = 50000.0
u = g.gen_test_signal(dur,dt,fmax,nc=10)

print 'creating filter..'
f1 = 10000.0
f2 = 20000.0
a = 1
#[numtaps,bands,desired,weight] = s.remezord([f1,f2],[1,0],[0.01,0.01],fs)
#b = remez(numtaps,bands,desired,weight)
b = firwin(50,f1/fs)
#print 'filtering with lfilter..'
#u_lfilter = lfilter(b,a,u)

print 'filtering with fftfilt..'
u_fftfilt = s.fftfilt(b,u)

