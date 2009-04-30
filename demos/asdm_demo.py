#!/usr/bin/env python

"""
Demos for ASDM time encoding and decoding algorithms.
"""

import sys

import numpy as np
import pylab as p

import bionet.utils.gen_test_signal as g
import bionet.utils.test_utils as tu
import bionet.tdc.asdm as a

# Define algorithm parameters and input signal:
dur = 0.1
dt = 1e-6
f = 32
bw = 2*np.pi*f
t = np.linspace(0,dur,int(dur/dt))

np.random.seed(0)

noise_power = None

if noise_power == None:
    fig_title = 'ASDM input signal with no noise'
else:
    fig_title = 'ASDM input signal with %d dB of noise' % noise_power
print fig_title
u = tu.func_timer(g.gen_test_signal)(dur,dt,f,noise_power)
tu.plot_signal(t,u,fig_title,'asdm_input.png')

b = 3.5  # bias
d = 0.7  # threshold
k = 0.01 # scaling factor

try:
    a.asdm_recoverable(u,bw,b,d,k)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

out_count = 0
out_count += 1
fig_title = 'encoding using ASDM algorithm'
print fig_title
s = tu.func_timer(a.asdm_encode)(u,dt,b,d,k)
tu.plot_encoded(t,u,s,fig_title,'asdm_output_%i.png' % out_count)

out_count += 1
fig_title = 'decoding using ASDM algorithm'
print fig_title
u_rec = tu.func_timer(a.asdm_decode)(s,dur,dt,bw,b,d,k)
tu.plot_compare(t,u,u_rec,fig_title,'asdm_output_%i.png' % out_count)

out_count += 1
fig_title = 'decoding using threshold-insensitive ASDM algorithm'
print fig_title
u_rec_ins = tu.func_timer(a.asdm_decode_ins)(s,dur,dt,bw,b)
tu.plot_compare(t,u,u_rec_ins,fig_title,'asdm_output_%i.png' % out_count)

