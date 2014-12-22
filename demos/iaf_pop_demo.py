#!/usr/bin/env python

"""
Demos of encoding and decoding algorithms using populations of
IAF neurons.
"""

# Copyright (c) 2009-2014, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

import sys
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
output_name = 'iaf_pop_demo_'
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

try:
    iaf.iaf_recoverable(u, bw, b1, d1, R1, C1)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

b2 = 3.4   # bias
d2 = 0.8   # threshold
R2 = 9.0   # resistance
C2 = 0.01  # capacitance

try:
    iaf.iaf_recoverable(u, bw, b2, d2, R2, C2)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

b_list = np.array([b1, b2])
d_list = np.array([d1, d2])
R_list = np.array([R1, R2])
C_list = np.array([C1, C2])

output_count += 1
fig_title = 'Signal Encoded Using Leaky IAF Encoder'
print fig_title
s_list = func_timer(iaf.iaf_encode_pop)([u, u], dt, b_list, d_list, R_list, C_list)
pl.plot_encoded(t, u, s_list[0], fig_title + ' #1',
                output_name + str(output_count) + output_ext)
output_count += 1
pl.plot_encoded(t, u, s_list[1], fig_title + ' #2',
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Leaky IAF Population Decoder'
print fig_title
u_rec = func_timer(iaf.iaf_decode_pop)(s_list, dur, dt, bw,
                                          b_list, d_list, R_list,
                                          C_list)
pl.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)

# Test ideal IAF algorithms:

b1 = 3.5     # bias
d1 = 0.7     # threshold
R1 = np.inf  # resistance
C1 = 0.01    # capacitance

try:
    iaf.iaf_recoverable(u, bw, b1, d1, R1, C1)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

b2 = 3.4     # bias
d2 = 0.8     # threshold
R2 = np.inf  # resistance
C2 = 0.01    # capacitance

try:
    iaf.iaf_recoverable(u, bw, b2, d2, R2, C2)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

b_list = [b1, b2]
d_list = [d1, d2]
R_list = [R1, R2]
C_list = [C1, C2]

output_count += 1
fig_title = 'Signal Encoded Using Ideal IAF Encoder'
print fig_title
s_list = func_timer(iaf.iaf_encode_pop)([u, u], dt, b_list, d_list, R_list, C_list)
pl.plot_encoded(t, u, s_list[0], fig_title + ' #1',
                output_name + str(output_count) + output_ext)
output_count += 1
pl.plot_encoded(t, u, s_list[1], fig_title + ' #2',
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Ideal IAF Population Decoder'
print fig_title
u_rec = func_timer(iaf.iaf_decode_pop)(s_list, dur, dt, bw,
                                          b_list, d_list, R_list,
                                          C_list)
pl.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)

