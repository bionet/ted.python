#!/usr/bin/env python

"""
Demos for real-time IAF time encoding and decoding algorithms.
"""

import sys
import numpy as np

import bionet.utils.gen_test_signal as g
import bionet.utils.test_utils as tu
import bionet.ted.iaf as iaf
import bionet.ted.rt as rt

# For determining output plot file names:
output_name = 'iaf_rt_demo_'
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
    fig_title = 'IAF input signal with no noise'
else:
    fig_title = 'IAF input signal with %d dB of noise' % noise_power
print fig_title
u = tu.func_timer(g.gen_test_signal)(dur, dt, f, noise_power)
tu.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)

b = 3.5   # bias
d = 0.7   # threshold
R = 10.0  # resistance
C = 0.01  # capacitance

# Define real time decoder stitching parameters:
N = 10
M = 2
K = 1

try:
    iaf.iaf_recoverable(u, bw, b, d, R, C)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

# Test leaky algorithm:

output_count += 1
fig_title = 'encoding using leaky real-time IAF algorithm'
print fig_title
encoder = rt.IAFRealTimeEncoder(dt, b, d, R, C)
s = tu.func_timer(encoder)(u)
tu.plot_encoded(t, u, s, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'decoding using leaky real-time IAF algorithm'
print fig_title
decoder = rt.IAFRealTimeDecoder(dt, bw, b, d, R, C, N, M, K)
u_rec = tu.func_timer(decoder)(s)
end = min(len(u), len(u_rec))
tu.plot_compare(t[:end], u[:end], u_rec[:end], fig_title,
                output_name + str(output_count) + output_ext)


# Test nonleaky algorithm:

R = np.inf

output_count += 1
fig_title = 'encoding using nonleaky real-time IAF algorithm'
print fig_title
encoder = rt.IAFRealTimeEncoder(dt, b, d, R, C)
s = tu.func_timer(encoder)(u)
tu.plot_encoded(t, u, s, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'decoding using nonleaky real-time IAF algorithm'
print fig_title
decoder = rt.IAFRealTimeDecoder(dt, bw, b, d, R, C, N, M, K)
u_rec = tu.func_timer(decoder)(s)
end = min(len(u), len(u_rec))
tu.plot_compare(t[:end], u[:end], u_rec[:end], fig_title,
                output_name + str(output_count) + output_ext)

