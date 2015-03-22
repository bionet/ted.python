#!/usr/bin/env python

"""
Demos for spline interpolation IAF population time encoding and
decoding algorithms.
"""

# Copyright (c) 2009-2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

import numpy as np

# Set matplotlib backend so that plots can be generated without a
# display:
import matplotlib
matplotlib.use('AGG')

from bionet.utils.misc import func_timer
import bionet.utils.band_limited as bl
import bionet.utils.plotting as pl
import bionet.ted.iaf as iaf

# For determining output plot file names:
output_name = 'iaf_pop_spline_demo_'
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
    fig_title = 'IAF Input Signal with No Noise'
else:
    fig_title = 'IAF Input Signal with %d dB of Noise' % noise_power
print fig_title
u = func_timer(bl.gen_band_limited)(dur, dt, f, noise_power)
pl.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)

# Test leaky IAF algorithms:

b1 = 3.5   # bias
d1 = 0.7   # threshold
R1 = 10.0  # resistance
C1 = 0.01  # capacitance

b2 = 3.4   # bias
d2 = 0.8   # threshold
R2 = 9.0   # resistance
C2 = 0.01  # capacitance

output_count += 1
fig_title = 'Signal Encoded Using Leaky IAF Encoder #1'
print fig_title
s1 = func_timer(iaf.iaf_encode)(u, dt, b1, d1, R1, C1)
pl.plot_encoded(t, u, s1, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Encoded Using Leaky IAF Encoder #2'
print fig_title
s2 = func_timer(iaf.iaf_encode)(u, dt, b2, d2, R2, C2)
pl.plot_encoded(t, u, s2, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Leaky\nSpline Interpolation IAF Population Decoder'
print fig_title
u_rec = func_timer(iaf.iaf_decode_spline_pop)([s1, s2], dur, dt,
                                                 [b1, b2], [d1, d2], [R1, R2],
                                                 [C1, C2])
pl.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)

# Test ideal IAF algorithms:

b1 = 3.5     # bias
d1 = 0.7     # threshold
R1 = np.inf  # resistance
C1 = 0.01    # capacitance

b2 = 3.4     # bias
d2 = 0.8     # threshold
R2 = np.inf  # resistance
C2 = 0.01    # capacitance

output_count += 1
fig_title = 'Signal Encoded Using Ideal IAF Encoder #1'
print fig_title
s1 = func_timer(iaf.iaf_encode)(u, dt, b1, d1, R1, C1)
pl.plot_encoded(t, u, s1, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Encoded Using Ideal IAF Encoder #2'
print fig_title
s2 = func_timer(iaf.iaf_encode)(u, dt, b2, d2, R2, C2)
pl.plot_encoded(t, u, s2, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Ideal\nSpline Interpolation IAF Population Decoder'
print fig_title
u_rec = func_timer(iaf.iaf_decode_spline_pop)([s1, s2], dur, dt,
                                                 [b1, b2], [d1, d2], [R1, R2],
                                                 [C1, C2])
pl.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)

