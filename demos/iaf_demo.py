#!/usr/bin/env python

"""
Demos for IAF time encoding and decoding algorithms.
"""

import sys
import numpy as np

import bionet.utils.gen_test_signal as g
import bionet.utils.test_utils as tu
import bionet.ted.iaf as a

# Define algorithm parameters and input signal:
dur = 0.1
dt = 1e-6
f = 32
bw = 2*np.pi*f
t = np.linspace(0, dur, int(dur/dt))

np.random.seed(0)

noise_power = None

if noise_power == None:
    fig_title = 'IAF input signal with no noise';
else:
    fig_title = 'IAF input signal with %d dB of noise' % noise_power;
print fig_title
u = tu.func_timer(g.gen_test_signal)(dur, dt, f, noise_power)
tu.plot_signal(t, u, fig_title,'iaf_input.png')

b = 3.5   # bias
d = 0.7   # threshold
R = 10.0  # resistance
C = 0.01  # capacitance

try:
    a.iaf_recoverable(u, bw, b, d, R, C)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

M = 5 # number of bins for fast decoding algorithm
L = 5 # number of recursions for recursive decoding algorithm

# Test leaky algorithms:
out_count = 0
out_count += 1
fig_title = 'encoding using leaky IAF algorithm'
print fig_title
s = tu.func_timer(a.iaf_encode)(u, dt, b, d, R, C, quad_method='rect')
tu.plot_encoded(t, u, s, fig_title,'iaf_output_%i.png' % out_count)

out_count += 1
fig_title = 'decoding using leaky IAF algorithm'
print fig_title
u_rec = tu.func_timer(a.iaf_decode)(s, dur, dt, bw, b, d, R, C)
tu.plot_compare(t, u, u_rec, fig_title,'iaf_output_%i.png' % out_count)

out_count += 1
fig_title = 'decoding using leaky fast IAF algorithm'
print fig_title
u_rec = tu.func_timer(a.iaf_decode_fast)(s, dur, dt, bw, M, b, d, R, C)
tu.plot_compare(t, u, u_rec, fig_title,'iaf_output_%i.png' % out_count)

# Test nonleaky algorithms:
R = np.inf

out_count += 1
fig_title = 'encoding using nonleaky IAF algorithm'
print fig_title
s = tu.func_timer(a.iaf_encode)(u, dt, b, d, R, C,quad_method='rect')
tu.plot_encoded(t, u, s, fig_title,'iaf_output_%i.png' % out_count)

out_count += 1
fig_title = 'decoding using nonleaky IAF algorithm'
print fig_title
u_rec = tu.func_timer(a.iaf_decode)(s, dur, dt, bw, b, d, R, C)
tu.plot_compare(t, u, u_rec, fig_title,'iaf_output_%i.png' % out_count)

out_count += 1
fig_title = 'decoding using nonleaky fast IAF algorithm'
print fig_title
u_rec = tu.func_timer(a.iaf_decode_fast)(s, dur, dt, bw, M, b, d, R, C)
tu.plot_compare(t, u, u_rec, fig_title,'iaf_output_%i.png' % out_count)

