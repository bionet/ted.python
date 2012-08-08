#!/usr/bin/env python

"""
Demos of real-time time encoding and decoding algorithms that use
IAF neurons.
"""

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
    fig_title = 'IAF Input Signal with No Noise'
else:
    fig_title = 'IAF Input Signal with %d dB of Noise' % noise_power
print fig_title
u = func_timer(bl.gen_band_limited)(dur, dt, f, noise_power)
pl.plot_signal(t, u, fig_title,
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
fig_title = 'Signal Encoded Using Leaky Real-Time IAF Encoder'
print fig_title
encoder = rt.IAFRealTimeEncoder(dt, b, d, R, C)
s = func_timer(encoder)(u)
pl.plot_encoded(t, u, s, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Leaky Real-Time IAF Decoder'
print fig_title
decoder = rt.IAFRealTimeDecoder(dt, bw, b, d, R, C, N, M, K)
u_rec = func_timer(decoder)(s)
end = min(len(u), len(u_rec))
pl.plot_compare(t[:end], u[:end], u_rec[:end], fig_title,
                output_name + str(output_count) + output_ext)


# Test ideal algorithm:

R = np.inf

output_count += 1
fig_title = 'Signal Encoded Using Ideal Real-Time IAF Encoder'
print fig_title
encoder = rt.IAFRealTimeEncoder(dt, b, d, R, C)
s = func_timer(encoder)(u)
pl.plot_encoded(t, u, s, fig_title,
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Ideal Real-Time IAF Decoder'
print fig_title
decoder = rt.IAFRealTimeDecoder(dt, bw, b, d, R, C, N, M, K)
u_rec = func_timer(decoder)(s)
end = min(len(u), len(u_rec))
pl.plot_compare(t[:end], u[:end], u_rec[:end], fig_title,
                output_name + str(output_count) + output_ext)

