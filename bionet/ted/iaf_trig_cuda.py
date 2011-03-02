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
from numpy import ceil, cumsum, inf, isinf, pi, real

import scikits.cuda.misc as cumisc
import scikits.cuda.linalg as culinalg

from iaf_cuda import iaf_encode
from prodtrans import prodtrans

# Pseudoinverse singular value cutoff:
__pinv_rcond__ = 1e-8

compute_F_template = Template("""
#include <pycuda/pycuda-complex.hpp>

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#endif

#define EM(m,t,bw,M) exp(COMPLEX(0, m*bw*t/M))

// s: interspike intervals
// ts: spike times
// F: computed F matrix
// bw: bandwidth (rad/s)
// R: resistance
// C: capacitance
// M: trigonometric polynomial order
// N: (len(s)-1)*(2*M+1)
__global__ void compute_F_ideal(FLOAT *s, FLOAT *ts, COMPLEX *F, FLOAT bw, int M,
                                unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    unsigned int k = idx/(2*M+1);
    int m = (idx%(2*M+1))-M;
    
    if (idx < N) {
        if (m == 0) {
            F[idx] = s[k+1];
        } else {
            F[idx] = conj((EM(-m,ts[k+1],bw,M)-EM(-m,ts[k],bw,M))/COMPLEX(0,-m*bw/M));
        }
    }
}                          

__global__ void compute_F_leaky(FLOAT *s, FLOAT *ts, COMPLEX *F, FLOAT bw,
                                FLOAT R, FLOAT C, int M,
                                unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    unsigned int k = idx/(2*M+1);
    int m = (idx%(2*M+1))-M;
    
    FLOAT RC = R*C;
    if (idx < N) {
        F[idx] = conj((RC*EM(-m,ts[k+1],bw,M)-
                      RC*exp(-s[k+1]/RC)*EM(-m,ts[k],bw,M))/COMPLEX(1,-m*bw*RC/M));
    }
}                          

""")

compute_q_template = Template("""
#include <pycuda/pycuda-complex.hpp>

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#endif

// s: interspike intervals
// q: computed q array
// b: bias
// d: threshold
// R: resistance
// C: capacitance
// N: (len(s)-1)
__global__ void compute_q_ideal(FLOAT *s, COMPLEX *q, FLOAT b,
                                FLOAT d, FLOAT C, unsigned int N) {                          
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < N) {
        q[idx] = COMPLEX(C*d-b*s[idx+1]);
    }
}

__global__ void compute_q_leaky(FLOAT *s, COMPLEX *q, FLOAT b,
                                FLOAT d, FLOAT R, FLOAT C, unsigned int N) {                          
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < N) {
        q[idx] = COMPLEX(C*(d+b*R*(exp(-s[idx+1]/(R*C))-1)));
    }
}
""")

compute_u_template = Template("""
#include <pycuda/pycuda-complex.hpp>
 
#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#endif

#define EM(m,t,bw,M) exp(COMPLEX(0, m*bw*t/M))

// u_rec: reconstructed signal
// c: reconstruction coefficients
// bw: bandwidth (rad/s)
// dt: time resolution of reconstructed signal
// M: trigonometric polynomial order
// Nt: len(t)
__global__ void compute_u(COMPLEX *u_rec, COMPLEX *c,
                          FLOAT bw, FLOAT dt, int M, unsigned Nt) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
                       
    if (idx < Nt) {
        COMPLEX u_temp = COMPLEX(0);        
        for (int m = -M; m <= M+1; m++) {
            u_temp += c[m+M]*EM(m,idx*dt,bw,M);
        }
        u_rec[idx] = u_temp;
    }
}
""")

def iaf_decode_trig(s, dur, dt, bw, b, d, R=inf, C=1.0, M=5, smoothing=0.0):
    """
    IAF time decoding machine.
    
    Decode a finite length signal encoded with an Integrate-and-Fire
    neuron.

    Parameters
    ----------
    s : ndarray of floats
        Encoded signal. The values represent the time between spikes (in s).
    dur : float
        Duration of signal (in s).
    dt : float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw : float
        Signal bandwidth (in rad/s).
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.
    M : int
        2*M+1 coefficients are used for reconstructing the signal.
    smoothing : float
        Smoothing parameter.
        
    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.

    """

    N = len(s)
    float_type = s.dtype.type
    if float_type == np.float32:
        use_double = 0
        complex_type = np.complex64
    else:
        use_double = 1
        complex_type = np.complex128
        
    T = 2*pi*M/bw
    if T < dur:
        raise ValueError('2*pi*M/bw must exceed the signal length')

    dev = cumisc.get_current_device()
    
    # Get device constraints:
    max_threads_per_block, max_block_dim, max_grid_dim = cumisc.get_dev_attrs(dev)
    max_blocks_per_grid = max(max_grid_dim)

    max_threads_per_block = 256
    
    # Prepare kernels:
    cache_dir = None
    compute_q_mod = \
                  SourceModule(compute_q_template.substitute(use_double=use_double,
                               max_threads_per_block=max_threads_per_block,
                               max_blocks_per_grid=max_blocks_per_grid),
                               cache_dir=cache_dir)
    compute_q_ideal = compute_q_mod.get_function('compute_q_ideal')
    compute_q_leaky = compute_q_mod.get_function('compute_q_leaky')

    compute_F_mod = \
                  SourceModule(compute_F_template.substitute(use_double=use_double,
                               max_threads_per_block=max_threads_per_block,
                               max_blocks_per_grid=max_blocks_per_grid),
                               cache_dir=cache_dir)
    compute_F_ideal = compute_F_mod.get_function('compute_F_ideal')
    compute_F_leaky = compute_F_mod.get_function('compute_F_leaky')

    compute_u_mod = \
                  SourceModule(compute_u_template.substitute(use_double=use_double,
                               max_threads_per_block=max_threads_per_block,
                               max_blocks_per_grid=max_blocks_per_grid),
                               cache_dir=cache_dir)
    compute_u = compute_u_mod.get_function('compute_u')

    # Load data into GPU memory:
    s_gpu = gpuarray.to_gpu(s)

    # XXX: Eventually replace this with a PyCUDA equivalent
    ts = cumsum(s)
    ts_gpu = gpuarray.to_gpu(ts)

    # Set up GPUArrays for intermediary data. Note that all of the
    # arrays are complex to facilitate use of CUBLAS matrix
    # multiplication functions:
    q_gpu = gpuarray.empty((N-1, 1), complex_type)
    F_gpu = gpuarray.empty((N-1, 2*M+1), complex_type)

    # Get required block/grid sizes:
    block_dim_s, grid_dim_s = cumisc.select_block_grid_sizes(dev,
                                                             q_gpu.shape,
                                                             max_threads_per_block)
    block_dim_F, grid_dim_F = cumisc.select_block_grid_sizes(dev,
                                                             F_gpu.shape,
                                                             max_threads_per_block)
    if isinf(R):
        compute_q_ideal(s_gpu, q_gpu, float_type(b), float_type(d),
                        float_type(C), np.uint32(N-1),
                        block=block_dim_s, grid=grid_dim_s)
        compute_F_ideal(s_gpu, ts_gpu, F_gpu, float_type(bw),
                        np.int32(M), np.uint32((N-1)*(2*M+1)),
                        block=block_dim_F, grid=grid_dim_F)
    else:
        compute_q_leaky(s_gpu, q_gpu, float_type(b), float_type(d),
                        float_type(R), float_type(C), np.uint32(N-1),
                        block=block_dim_s, grid=grid_dim_s)
        compute_F_leaky(s_gpu, ts_gpu, F_gpu, float_type(bw),
                        float_type(R), float_type(C),
                        np.int32(M), np.uint32((N-1)*(2*M+1)),
                        block=block_dim_F, grid=grid_dim_F)

    # Compute the product of F^H and q first so that both F^H and q
    # can be dropped from memory:
    FH_gpu = culinalg.hermitian(F_gpu)
    FHq_gpu = culinalg.dot(FH_gpu, q_gpu)
    del FH_gpu, q_gpu
    
    if smoothing == 0:
        c_gpu = culinalg.dot(culinalg.pinv(prodtrans(F_gpu),
                                           __pinv_rcond__),
                             FHq_gpu)
    else:
        c_gpu = culinalg.dot(culinalg.pinv(prodtrans(F_gpu)+
                                           (N-1)*smoothing*culinalg.eye(2*M+1,
                                                                        float_type),
                                           __pinv_rcond__),
                             FHq_gpu)
        
    # Allocate array for reconstructed signal:
    Nt = int(ceil(dur/dt))
    u_rec_gpu = gpuarray.zeros(Nt, complex_type)

    # Get required block/grid sizes:
    block_dim_t, grid_dim_t = \
                 cumisc.select_block_grid_sizes(dev, Nt, max_threads_per_block)

    # Reconstruct signal:
    compute_u(u_rec_gpu, c_gpu, float_type(bw),
              float_type(dt),
              np.int32(M),
              np.uint32(Nt),
              block=block_dim_t, grid=grid_dim_t)

    return real(u_rec_gpu.get())

