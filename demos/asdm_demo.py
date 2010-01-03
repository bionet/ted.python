#!/usr/bin/env python

"""
Demos of basic time encoding and decoding algorithms that use
asynchronous sigma-delta modulators.
"""

import sys
import numpy as np

import bionet.utils.gen_test_signal as g
import bionet.utils.test_utils as tu
import bionet.ted.asdm as asdm

# For determining output plot file names:
output_name = 'asdm_demo_'
output_count = 0
output_ext = '.png'

# Define algorithm parameters and input signal:
dur = 0.1
dt = 1e-6
f = 32
bw = 2*np.pi*f
t = np.arange(0, dur, dt)

np.random.seed(0)

noise_power = None
if noise_power == None:
    fig_title = 'ASDM input signal with no noise'
else:
    fig_title = 'ASDM input signal with %d dB of noise' % noise_power
print fig_title
u = tu.func_timer(g.gen_test_signal)(dur, dt, f, noise_power)
tu.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)

b = 3.5  # bias
d = 0.7  # threshold
k = 0.01 # scaling factor

M = 5    # number of bins for fast decoding algorithm

try:
    asdm.asdm_recoverable(u, bw, b, d, k)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

output_count += 1
fig_title = 'encoding using ASDM algorithm'
print fig_title
s = tu.func_timer(asdm.asdm_encode)(u, dt, b, d, k)
tu.plot_encoded(t, u, s, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'decoding using ASDM algorithm'
print fig_title
u_rec = tu.func_timer(asdm.asdm_decode)(s, dur, dt, bw, b, d, k)
tu.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'decoding using threshold-insensitive ASDM algorithm'
print fig_title
u_rec_ins = tu.func_timer(asdm.asdm_decode_ins)(s, dur, dt, bw, b)
tu.plot_compare(t, u, u_rec_ins, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'decoding using fast ASDM algorithm'
print fig_title
u_rec = tu.func_timer(asdm.asdm_decode_fast)(s, dur, dt, bw, M, b, d, k)
tu.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)
