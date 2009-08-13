#!/usr/bin/env python

"""
Demos for real-time ASDM time encoding and decoding algorithms.
"""

import sys
import numpy as np

import bionet.utils.gen_test_signal as g
import bionet.utils.test_utils as tu
import bionet.ted.asdm as a
import bionet.ted.rt as rt

# Define algorithm parameters and input signal:
dur = 0.1
dt = 1e-6
f = 32
bw = 2*np.pi*f
t = np.linspace(0, dur, int(dur/dt))

np.random.seed(0)

noise_power = None

if noise_power == None:
    fig_title = 'ASDM input signal with no noise'
else:
    fig_title = 'ASDM input signal with %d dB of noise' % noise_power
print fig_title
u = tu.func_timer(g.gen_test_signal)(dur, dt, f, noise_power)
tu.plot_signal(t, u, fig_title, 'asdm_real_input.png')

b = 3.5  # bias
d = 0.7  # threshold
k = 0.01 # scaling factor

N = 10;  # real time decoder stitching parameters
M = 2;
K = 1;

try:
    a.asdm_recoverable(u, bw, b, d, k)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

out_count = 0
out_count += 1
fig_title = 'encoding using real-time ASDM algorithm'
print fig_title
encoder = rt.ASDMRealTimeEncoder(dt, b, d, k)
s = tu.func_timer(encoder)(u)
tu.plot_encoded(t, u, s, fig_title, 'asdm_real_output_%i.png' % out_count)

out_count += 1
fig_title = 'decoding using real-time algorithm'
print fig_title
decoder = rt.ASDMRealTimeDecoder(dt, bw, b, d, k, N, M, K)
u_rec = tu.func_timer(decoder)(s)
end = min(len(u), len(u_rec))
tu.plot_compare(t[:end], u[:end], u_rec[:end],
                fig_title, 'asdm_real_output_%i.png' % out_count)

out_count += 1
fig_title = 'decoding using real-time threshold-insensitive algorithm'
print fig_title
decoder = rt.ASDMRealTimeDecoderIns(dt, bw, b, N, M, K)
u_rec_ins = tu.func_timer(decoder)(s)
end = min(len(u), len(u_rec_ins))
tu.plot_compare(t[:end], u[:end], u_rec_ins[:end],
                fig_title, 'asdm_real_output_%i.png' % out_count)

