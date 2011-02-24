#!/usr/bin/env python

"""
Demos of encoding and decoding algorithms using populations of
IAF neurons.
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
import bionet.ted.iaf_pop_trig_cuda as iaf_pop_trig_cuda

# Get the automatically selected GPU device:
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

import scikits.cuda.autoinit
#import scikits.cuda.misc as cumisc
#cumisc.init_device(0)
import scikits.cuda.linalg as culinalg

# For determining output plot file names:
output_name = 'iaf_pop_trig_cuda_demo_'
output_count = 0
output_ext = '.png'

# Define algorithm parameters and input signal:
dur = 0.1
dt = 1e-6
f = 64
bw = 2*np.pi*f
t = np.arange(0, dur, dt)

np.random.seed(0)

noise_power = None
if noise_power == None:
    fig_title = 'IAF Input Signal with No Noise'
else:
    fig_title = 'IAF Input Signal with %d dB of Noise' % noise_power
print fig_title
u = func_timer(g.gen_test_signal)(dur, dt, f, noise_power)
pl.plot_signal(t, u, fig_title,
               output_name + str(output_count) + output_ext)

# Trigonometric polynomial order:
M = 75

# Test leaky IAF algorithms:

b1 = 7.5   # bias
d1 = 0.7   # threshold
R1 = 10.0  # resistance
C1 = 0.01  # capacitance

try:
    iaf.iaf_recoverable(u, bw, b1, d1, R1, C1)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

b2 = 6.4   # bias
d2 = 0.8   # threshold
R2 = 9.0   # resistance
C2 = 0.01  # capacitance

try:
    iaf.iaf_recoverable(u, bw, b2, d2, R2, C2)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

u_gpu = gpuarray.to_gpu(u)
b_gpu = gpuarray.to_gpu(np.array([b1, b2]))
d_gpu = gpuarray.to_gpu(np.array([d1, d2]))
R_gpu = gpuarray.to_gpu(np.array([R1, R2]))
C_gpu = gpuarray.to_gpu(np.array([C1, C2]))

output_count += 1
fig_title = 'Signal Encoded Using Leaky IAF Encoder'
print fig_title
s_gpu, ns_gpu = func_timer(iaf_pop_trig_cuda.iaf_encode_pop)(u_gpu,
                                                             dt, b_gpu, d_gpu, R_gpu, C_gpu)
s = s_gpu.get()
ns = ns_gpu.get()
pl.plot_encoded(t, u, s[0,0:ns[0]], fig_title + ' #1',
                output_name + str(output_count) + output_ext)
output_count += 1
pl.plot_encoded(t, u, s[1,0:ns[1]], fig_title + ' #2',
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Leaky IAF Population Decoder'
print fig_title
u_rec = func_timer(iaf_pop_trig_cuda.iaf_decode_pop)(s_gpu, ns_gpu, dur, dt, bw,
                                       b_gpu, d_gpu, R_gpu,
                                       C_gpu, M)
pl.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)

# Test ideal IAF algorithms:

b1 = 7.5     # bias
d1 = 0.7     # threshold
R1 = np.inf  # resistance
C1 = 0.01    # capacitance

try:
    iaf.iaf_recoverable(u, bw, b1, d1, R1, C1)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

b2 = 6.4     # bias
d2 = 0.8     # threshold
R2 = np.inf  # resistance
C2 = 0.01    # capacitance

try:
    iaf.iaf_recoverable(u, bw, b2, d2, R2, C2)
except ValueError('reconstruction condition not satisfied'):
    sys.exit()

u_gpu = gpuarray.to_gpu(u)
b_gpu = gpuarray.to_gpu(np.array([b1, b2]))
d_gpu = gpuarray.to_gpu(np.array([d1, d2]))
R_gpu = gpuarray.to_gpu(np.array([R1, R2]))
C_gpu = gpuarray.to_gpu(np.array([C1, C2]))

output_count += 1
fig_title = 'Signal Encoded Using Ideal IAF Encoder'
print fig_title
s_gpu, ns_gpu = func_timer(iaf_pop_trig_cuda.iaf_encode_pop)(u_gpu, dt, b_gpu, d_gpu, R_gpu, C_gpu)
s = s_gpu.get()
ns = ns_gpu.get()
pl.plot_encoded(t, u, s[0,0:ns[0]], fig_title + ' #1',
                output_name + str(output_count) + output_ext)
output_count += 1
pl.plot_encoded(t, u, s[1,0:ns[1]], fig_title + ' #2',
                output_name + str(output_count) + output_ext)

output_count += 1
fig_title = 'Signal Decoded Using Ideal IAF Population Decoder'
print fig_title
u_rec = func_timer(iaf_pop_trig_cuda.iaf_decode_pop)(s_gpu, ns_gpu, dur, dt, bw,
                                                     b_gpu, d_gpu,
                                                     R_gpu, C_gpu, M)
pl.plot_compare(t, u, u_rec, fig_title,
                output_name + str(output_count) + output_ext)

# s_list = [s[0, 0:ns[0]], s[1, 0:ns[1]]]
# b_list = b_gpu.get().tolist()
# d_list = d_gpu.get().tolist()
# R_list = R_gpu.get().tolist()
# C_list = C_gpu.get().tolist()
# import bionet.ted.iaf_trig as iaf_trig
# u_rec2 = iaf_trig.iaf_decode_pop(s_list, dur, dt, bw, b_list,
#                                  d_list, R_list, C_list)
