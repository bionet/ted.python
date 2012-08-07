#!/usr/bin/env python

"""
Demos for basic time encoding and decoding algorithms that use
IAF neurons. The decoding algorithms assume a trigonometric polynomial
approximation of the input signals.
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
import bionet.ted.iaf as iaf
import bionet.ted.iaf_trig as iaf_trig

# For determining output plot file names:
output_name = 'iaf_trig_demo_'
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
    fig_title = 'IAF Input Signal with no Noise';
else:
    fig_title = 'IAF Input Signal with %d dB of Noise' % noise_power;
print fig_title
u = func_timer(g.gen_test_signal)(dur, dt, f, noise_power)
pl.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)

b = 3.5   # bias
d = 0.7   # threshold
R = 10.0  # resistance
C = 0.01  # capacitance

try:
    iaf.iaf_recoverable(u, bw, b, d, R, C)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

# Trigonometric polynomial order:
M = 31

# Test leaky algorithms:

output_count += 1
fig_title = 'Signal Encoded Using Leaky IAF Encoder'
print fig_title
s = func_timer(iaf.iaf_encode)(u, dt, b, d, R, C)
pl.plot_encoded(t, u, s, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Leaky Trigonometric IAF Decoder'
print fig_title
u_rec = func_timer(iaf_trig.iaf_decode)(s, dur, dt, bw, b, d, R, C, M)     
pl.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)

# Test ideal algorithms:

R = np.inf

output_count += 1
fig_title = 'Signal Encoded Using Ideal IAF Encoder'
print fig_title
s = func_timer(iaf.iaf_encode)(u, dt, b, d, R, C)
pl.plot_encoded(t, u, s, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Ideal Trigonometric IAF Decoder'
print fig_title
u_rec = func_timer(iaf_trig.iaf_decode)(s, dur, dt, bw, b, d, R, C, M)
pl.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)

