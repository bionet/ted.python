#!/usr/bin/env python

"""
Demos of MIMO time encoding and decoding algorithms that use IAF
neurons with delays.
"""

import numpy as np

# Set matplotlib backend so that plots can be generated without a
# display:
import matplotlib
matplotlib.use('AGG')

from bionet.utils.misc import func_timer
import bionet.utils.gen_test_signal as g
import bionet.utils.plotting as pl
import bionet.ted.iaf as iaf

# For determining output plot file names:
output_name = 'iaf_delay_demo_'
output_count = 0
output_ext = '.png'

# Define input signal generation parameters:
T = 0.05
dur = 2*T
dt = 1e-6
f = 100
bw = 2*np.pi*f
t = np.arange(0, T, dt)

np.random.seed(0)

noise_power = None
comps = 10

if noise_power == None:
    fig_title = 'IAF Input Signal with No Noise'
else:
    fig_title = 'IAF Input Signal with %d dB of Noise' % noise_power

M = 3 # number of input signals
N = 9 # number of neurons

u_list = []
for i in xrange(M):
    fig_title_in = fig_title + ' (Signal #' + str(i+1) + ')'
    print fig_title_in
    u = func_timer(g.gen_test_signal)(dur, dt, f, noise_power, comps)
    u /= max(u)
    u *= 1.5
    pl.plot_signal(t, u[0:len(t)], fig_title_in,
                   output_name + str(output_count) + output_ext)
    u_list.append(u)
    output_count += 1

# Define neuron parameters:
def randu(a, b, *d):
    """Create an array of the given shape and propagate it with random
    samples from a uniform distribution over ``[a, b)``."""

    if a >= b:
        raise ValueError('b must exceed a')
    return a+(b-a)*np.random.rand(*d)

b_list = list(randu(2.3, 3.3, N))
d_list = list(randu(0.2, 0.3, N))
k_list = list(0.01*np.ones(N))
a_list = map(list, np.reshape(np.random.exponential(0.003, N*M), (N, M)))
w_list = map(list, np.reshape(randu(0.5, 1.0, N*M), (N, M)))

fig_title = 'Signal Encoded Using Delayed IAF Encoder'
print fig_title
s_list = func_timer(iaf.iaf_encode_delay)(u_list, T, dt, b_list, d_list,
                                           k_list, a_list, w_list)

for i in xrange(M):
    for j in xrange(N):
        fig_title_out = fig_title + '\n(Signal #' + str(i+1) + \
                        ', Neuron #' + str(j+1) + ')'
        pl.plot_encoded(t, u_list[i][0:len(t)], s_list[j], fig_title_out,
                        output_name + str(output_count) + output_ext)
        output_count += 1
    
fig_title = 'Signal Decoded Using Delayed IAF Decoder'
print fig_title
u_rec_list = func_timer(iaf.iaf_decode_delay)(s_list, T, dt, b_list, d_list, k_list,
                                                 a_list, w_list)

for i in xrange(M):
    fig_title_out = fig_title + ' (Signal #' + str(i+1) + ')'
    pl.plot_compare(t, u_list[i][0:len(t)], u_rec_list[i], fig_title_out, 
                    output_name + str(output_count) + output_ext)
    output_count += 1
