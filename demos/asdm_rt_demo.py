#!/usr/bin/env python

"""
Demos for real-time ASDM time encoding and decoding algorithms.
"""

import sys
import numpy as np

# Set matplotlib backend so that plots can be generated without a
# display:
import matplotlib
matplotlib.use('AGG')

import bionet.utils.gen_test_signal as g
import bionet.utils.test_utils as tu
import bionet.ted.asdm as asdm
import bionet.ted.rt as rt

# For determining output plot file names:
output_name = 'asdm_rt_demo_'
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
u = tu.func_timer(g.gen_test_signal)(dur, dt, f, noise_power)
tu.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)

# Define encoding parameters:
dte = dt
quad_method = 'trapz'

b = 3.5   # bias
d = 0.7   # threshold
k = 0.01  # scaling factor

# Define real time decoder stitching parameters:
N = 10
M = 2
K = 1

try:
    asdm.asdm_recoverable(u, bw, b, d, k)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

output_count += 1
fig_title = 'Signal Encoded Using Real-Time ASDM Encoder'
print fig_title
encoder = rt.ASDMRealTimeEncoder(dt, b, d, k)
s = tu.func_timer(encoder)(u)
tu.plot_encoded(t, u, s, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Real-Time ASDM Decoder'
print fig_title
decoder = rt.ASDMRealTimeDecoder(dt, bw, b, d, k, N, M, K)
u_rec = tu.func_timer(decoder)(s)
end = min(len(u), len(u_rec))
tu.plot_compare(t[:end], u[:end], u_rec[:end], fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Real-Time\nThreshold-Insensitive ASDM Decoder'
print fig_title
decoder = rt.ASDMRealTimeDecoderIns(dt, bw, b, N, M, K)
u_rec_ins = tu.func_timer(decoder)(s)
end = min(len(u), len(u_rec_ins))
tu.plot_compare(t[:end], u[:end], u_rec_ins[:end], fig_title,
                output_name + str(output_count) + output_ext)

