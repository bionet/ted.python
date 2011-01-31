#!/usr/bin/env python

"""
Time decoding algorithms that use the trigonometric polynomial
approximation.

These functions make use of CUDA.

"""

from string import Template
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np
from numpy import inf


# The kernel templates
# compute_q_ideal_mod_template and
# compute_ts_ideal_mod_template from above are reused here.
compute_F_ideal_trig_mod_template = Template("""
#include <cuComplexFuncs.h>

#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define FLOAT float
#define CONJ(z) conjf(z)
#define EXP(z) cuCexpf(z)
#else
#define FLOAT double
#define CONJ(z) conj(z)
#define EXP(x) cuCexp(z)
#endif

__device__ FLOAT em(int m, FLOAT t, FLOAT bw, int M) {
    return EXP(make_cuFloatComplex(0, m*bw*t/M));
}

// N must equal the square of one less than the length of ts:
__global__ void compute_F(FLOAT *s, FLOAT *ts, FLOAT *F, FLOAT bw, int M,
                          unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    unsigned int k = idx/(2*M+1);
    unsigned int m = idx%(2*M+1);

    if (idx < N) {
        if (m == M+1) {
            G[idx] = s[k+1];
        } else {
            G[idx] = CONJ
        }
    }
}                          
""")


def iaf_decode_trig(s, dur, dt, bw, b, d, R=inf, C=1.0, M=5, dev=None):
    pass

