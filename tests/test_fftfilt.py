#!/usr/bin/env python

"""
Test fftfilt function.
"""

from numpy.testing import *
import numpy as np
import scipy.signal as si 
from unittest import main

import bionet.utils.signal_extras as s
import bionet.utils.gen_test_signal as g

class TestFFTFilt(TestCase):
    def setUp(self):
        dur = 0.2
        fmax = 50000.0
        dt = 1e-6
        self.fs = 1/dt
        self.u = g.gen_test_signal(dur,dt,fmax,nc=10)
        
    def test_fftfilt(self):
        f = 10000.0
        b = si.firwin(50,f/self.fs)
        
        u_lfilter = si.lfilter(b,1,self.u)
        u_fftfilt = s.fftfilt(b,self.u)
        assert_almost_equal(u_lfilter,u_fftfilt)

if __name__ == "__main__":
    main()
        
