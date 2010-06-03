#!/usr/bin/env python

"""
Generate plots for overview section of TED documentation.
"""

import sys
import numpy as np
import matplotlib as mp

import bionet.utils.gen_test_signal as g
import bionet.utils.test_utils as tu
import bionet.ted.iaf as iaf

# Set plot generation DPI:
mp.rc('savefig', dpi=80)
output_ext = '.png'
output_dir = 'images/'

# Define algorithm parameters and input signal:
dur = 0.1
dt = 1e-6
f = 32
bw = 2*np.pi*f
t = np.arange(0, dur, dt)

np.random.seed(0)

noise_power = None
fig_title = 'IAF input signal';
print fig_title
u = tu.func_timer(g.gen_test_signal)(dur, dt, f, noise_power)
tu.plot_signal(t, u, fig_title, output_dir + 'overview_input' + output_ext)

b = 3.5     # bias
d = 0.7     # threshold
R = np.inf  # resistance
C = 0.01    # capacitance

try:
    iaf.iaf_recoverable(u, bw, b, d, R, C)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

fig_title = 'Signal encoded by IAF neuron'
print fig_title
s = tu.func_timer(iaf.iaf_encode)(u, dt, b, d, R, C)
tu.plot_encoded(t, u, s, fig_title,
                output_dir + 'overview_encoded' + output_ext)

fig_title = 'Decoded signal and reconstruction error'
print fig_title
u_rec = tu.func_timer(iaf.iaf_decode)(s, dur, dt, bw, b, d, R, C)
tu.plot_compare(t, u, u_rec, fig_title,
                output_dir + 'overview_decoded' + output_ext)

