#!/usr/bin/env python

"""
Generate a bandlimited test signal.
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
import bionet.utils.plotting as pl
import bionet.utils.band_limited as bl

# For determing output plot file names:
output_name = 'gen_band_limited_demo_'
output_count = 0
output_ext = '.png'

print 'creating test signal..'
dur = 1.0
fs = 1e4
dt = 1.0/fs
f = 10
t = np.arange(0, dur, dt)

np.random.seed(0)
out_count = 0
fig_title = 'test signal with no noise'
print fig_title
u = func_timer(bl.gen_band_limited)(dur, dt, f)
pl.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)

np.random.seed(0)
output_count += 1
noise_power = 1
fig_title = 'test signal with %d dB of noise' % noise_power
print fig_title
u = func_timer(bl.gen_band_limited)(dur, dt, f, noise_power)
pl.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)

np.random.seed(0)
output_count += 1
noise_power = -5
fig_title = 'test signal with %d dB of noise' % noise_power
print fig_title
u = func_timer(bl.gen_band_limited)(dur, dt, f, noise_power)
pl.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)

np.random.seed(0)
output_count += 1
noise_power = -5
nc = 8
fig_title = 'test signal with %d dB of noise and %i components' % \
            (noise_power, nc)
print fig_title
u = func_timer(bl.gen_band_limited)(dur, dt, f, noise_power, nc)
pl.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)

np.random.seed(0)
output_count += 1
f = 50
noise_power = -10
nc = 20
fig_title = 'test signal with %d dB of noise and %i components' % \
            (noise_power, nc)
print fig_title
u = func_timer(bl.gen_band_limited)(dur, dt, f, noise_power, nc)
pl.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)
