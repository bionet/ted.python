#!/usr/bin/env python

"""
Demos of basic time encoding and decoding algorithms that use
asynchronous sigma-delta modulators.
"""

import sys
import numpy as np

# Set matplotlib backend so that plots can be generated without a
# display:
import matplotlib
matplotlib.use('AGG')

from bionet.utils.misc import func_timer
import bionet.utils.gen_test_signal as g
import bionet.utils.plotting as pl
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
    fig_title = 'ASDM Input Signal with No Noise'
else:
    fig_title = 'ASDM Input Signal with %d dB of Noise' % noise_power
print fig_title
u = func_timer(g.gen_test_signal)(dur, dt, f, noise_power)
pl.plot_signal(t, u, fig_title,
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
fig_title = 'Signal Encoded Using ASDM Encoder'
print fig_title
s = func_timer(asdm.asdm_encode)(u, dt, b, d, k)
pl.plot_encoded(t, u, s, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using ASDM Decoder'
print fig_title
u_rec = func_timer(asdm.asdm_decode)(s, dur, dt, bw, b, d, k)
pl.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Threshold-Insensitive ASDM Decoder'
print fig_title
u_rec_ins = func_timer(asdm.asdm_decode_ins)(s, dur, dt, bw, b)
pl.plot_compare(t, u, u_rec_ins, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Decoded Using Fast ASDM Decoder'
print fig_title
u_rec = func_timer(asdm.asdm_decode_fast)(s, dur, dt, bw, M, b, d, k)
pl.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)
