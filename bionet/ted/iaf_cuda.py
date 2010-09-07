#!/usr/bin/env python

"""
Time encoding and decoding algorithms that make use of the
integrate-and-fire neuron model.

These functions make use of CUDA.

"""

from string import Template
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np
from numpy import inf

import cuda_utils.linalg as culinalg
import cuda_utils.misc as cumisc

# Get installation location of C headers:
from cuda_utils import install_headers

# Pseudoinverse singular value cutoff:
__pinv_rcond__ = 1e-6

# Kernel template for performing ideal/leaky IAF time encoding using a
# single encoder:
iaf_encode_mod_template = Template("""
#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define FLOAT float
#define EXP(x) expf(x)
#else
#define FLOAT double
#define EXP(x) exp(x)
#endif

// Nu must contain the length of u:
__global__ void iaf_encode(FLOAT *u, FLOAT *s,
                           unsigned int *i_s_0, FLOAT dt,
                           FLOAT b, FLOAT d,
                           FLOAT R, FLOAT C,
                           FLOAT *y_0, FLOAT *interval_0,
                           unsigned int use_trapz,
                           unsigned int Nu)
{
    unsigned int idx = threadIdx.x;

    FLOAT y;
    FLOAT interval;
    unsigned int i_s;
    FLOAT RC = R*C;
    unsigned int last;
    
    if (idx == 0) {
        y = y_0[0];
        interval = interval_0[0];
        i_s = i_s_0[0];           // index into spike array

        // Use the exponential Euler method when the neuron resistance
        // is not infinite:
        if ((use_trapz == 1) && isinf(R))
            last = Nu-1;
        else
            last = Nu;

        for (unsigned int i = 0; i < last; i++) {
            if isinf(R)
                if (use_trapz == 1)
                    y += dt*(b+(u[i]+u[i+1])/2)/C;
                else
                    y += dt*(b+u[i])/C;
            else
                y = y*EXP(-dt/RC)+R*(1.0-EXP(-dt/RC))*(b+u[i]);
            interval += dt;
            if (y >= d) {
                s[i_s] = interval;
                interval = 0;
                y = 0;
                i_s++;
            }
        }

        // Save the integrator and interval values for the next
        // iteration:
        y_0[0] = y;
        interval_0[0] = interval;
        i_s_0[0] = i_s;
    }                     
}
""")

def iaf_encode(u, dt, b, d, R=inf, C=1.0, dte=0.0, y=0.0, interval=0.0,
               quad_method='trapz', full_output=False, dev=None):
    """
    IAF time encoding machine.

    Notes
    -----
    This function assumes that no context is active on the specified
    device.
    
    """

    # Input sanity check:
    if u.dtype == np.float32:
        use_double = 0
        np_type = np.float32
    elif u.dtype == np.float64:
        use_double = 1
        np_type = np.float64
    else:
        raise ValueError('unsupported data type')
    
    # Configure kernel:
    iaf_encode_mod = \
                   SourceModule(iaf_encode_mod_template.substitute(use_double=use_double)) 
    iaf_encode = iaf_encode_mod.get_function("iaf_encode")

    # XXX: A very long s array might cause memory problems:
    s = np.zeros(len(u), np_type)
    i_s_0 = np.zeros(1, np.uint32)
    y_0 = np.asarray([y], np_type)
    interval_0 = np.asarray([interval], np_type)
    iaf_encode(drv.In(u), drv.Out(s), drv.InOut(i_s_0),
               np_type(dt), np_type(b),
               np_type(d), np_type(R), np_type(C), 
               drv.InOut(y_0), drv.InOut(interval_0),
               np.uint32(True if quad_method == 'trapz' else False),
               np.uint32(len(u)),
               block=(1, 1, 1))

    if full_output:
        return s[0:i_s_0[0]], dt, b, d, R, C, y_0[0], interval_0[0], \
               quad_method, full_output
    else:
        return s[0:i_s_0[0]]

# Kernel template for computing q in the pseudoinverse-based time
# decoder of a signal encoded by an ideal IAF neuron:
compute_q_ideal_mod_template = Template("""
#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define FLOAT float
#else
#define FLOAT double
#endif

// N must equal one less the length of s:
__global__ void compute_q(FLOAT *s, FLOAT *q, FLOAT b,
                          FLOAT d, FLOAT C, unsigned int N) {                          
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < N) {
        q[idx] = C*d-b*s[idx+1];
    }
}
""")

# Kernel template for computing spike times:
compute_ts_ideal_mod_template = Template("""
#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define FLOAT float
#else
#define FLOAT double
#endif

// N == len(s)
__global__ void compute_ts(FLOAT *s, FLOAT *ts, unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < N) {
        ts[idx] = 0.0;
        for (unsigned int i = 0; i < idx+1; i++)
            ts[idx] += s[i]; 
    }
}
""")

# Kernel template for computing midpoints between spikes:
compute_tsh_ideal_mod_template = Template("""
#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define FLOAT float
#else
#define FLOAT double
#endif

// Nsh == len(ts)-1
__global__ void compute_tsh(FLOAT *ts, FLOAT *tsh, unsigned int Nsh) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;

    if (idx < Nsh) {
        tsh[idx] = (ts[idx]+ts[idx+1])/2;
    }
}   
""")

# Kernel template for computing the recovery matrix:
compute_G_ideal_mod_template = Template("""
#include <cuConstants.h>    // needed to provide PI
#include <cuSpecialFuncs.h> // needed to provide sici()

#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define FLOAT float
#define SICI(x, si, ci) sicif(x, si, ci)
#else
#define FLOAT double
#define SICI(x, si, ci) sici(x, si, ci)
#endif

// N must equal the square of one less than the length of ts:
__global__ void compute_G(FLOAT *ts, FLOAT *tsh, FLOAT *G, FLOAT bw, unsigned int N) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    unsigned int ix = idx/${cols};
    unsigned int iy = idx%${cols};
    FLOAT si0, si1, ci;
    
    if (idx < N) {
        SICI(bw*(ts[ix+1]-tsh[iy]), &si1, &ci);
        SICI(bw*(ts[ix]-tsh[iy]), &si0, &ci);
        G[idx] = (si1-si0)/PI;
    }
}
""")

# Kernel template for reconstructing the encoded signal:
compute_u_ideal_mod_template = Template("""
#include <cuConstants.h>    // needed to provide PI
#include <cuSpecialFuncs.h> // needed to provide sinc()

#define USE_DOUBLE ${use_double}
#if USE_DOUBLE == 0
#define FLOAT float
#define SINC(x) sincf(x)
#else
#define FLOAT double
#define SINC(x) sinc(x)
#endif

// Nt == len(t)
// Nsh == len(tsh)
__global__ void compute_u(FLOAT *u_rec, FLOAT *t, FLOAT *tsh, FLOAT *c,
                          FLOAT bw, unsigned Nt, unsigned int Nsh) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    FLOAT bwpi = bw/PI;
    FLOAT u_temp = 0;
    
    // Each thread reconstructs the signal at time t[idx]:
    if (idx < Nt) {
        for (unsigned int i = 0; i < Nsh; i++) {
            u_temp += SINC(bwpi*(t[idx]-tsh[i]))*bwpi*c[i];
        }
        u_rec[idx] = u_temp;
    }
}
""")

def iaf_decode(s, dur, dt, bw, b, d, R=inf, C=1.0, dev=None):

    # Use single precision for all of the computations:
    stype = np.float32
    if s.dtype != stype:
        raise ValueError('unsupported data type')
        
    if not np.isinf(R):
        raise ValueError('decoding for leaky neuron not implemented yet')

    # Count total number of spike intervals:
    N = len(s)
    
    # Get device constraints:
    max_threads_per_block, max_block_dim, max_grid_dim = cumisc.get_dev_attrs(dev)
    max_blocks_per_grid = max(max_grid_dim)

    # Use a smaller block size than the maximum to prevent the kernels
    # from using too many registers:
    max_threads_per_block = 256
    
    # Prepare kernels:
    compute_q_ideal_mod = \
                        SourceModule(compute_q_ideal_mod_template.substitute(use_double=0,
                                     max_threads_per_block=max_threads_per_block,
                                     max_blocks_per_grid=max_blocks_per_grid))
    compute_q_ideal = \
                    compute_q_ideal_mod.get_function('compute_q')

    compute_ts_ideal_mod = \
                         SourceModule(compute_ts_ideal_mod_template.substitute(use_double=0,
                                     max_threads_per_block=max_threads_per_block,
                                     max_blocks_per_grid=max_blocks_per_grid))    
    compute_ts_ideal = \
                     compute_ts_ideal_mod.get_function('compute_ts')

    compute_tsh_ideal_mod = \
                          SourceModule(compute_tsh_ideal_mod_template.substitute(use_double=0,
                                     max_threads_per_block=max_threads_per_block,
                                     max_blocks_per_grid=max_blocks_per_grid)) 
    compute_tsh_ideal = \
                      compute_tsh_ideal_mod.get_function('compute_tsh')
                          

    compute_G_ideal_mod = \
                        SourceModule(compute_G_ideal_mod_template.substitute(use_double=0,
                                     max_threads_per_block=max_threads_per_block,
                                     max_blocks_per_grid=max_blocks_per_grid,
                                     cols=(N-1)),
                                     options=['-I', install_headers])
    compute_G_ideal = compute_G_ideal_mod.get_function('compute_G') 

    compute_u_ideal_mod = \
                        SourceModule(compute_u_ideal_mod_template.substitute(use_double=0,
                                     max_threads_per_block=max_threads_per_block,
                                     max_blocks_per_grid=max_blocks_per_grid),
                                     options=["-I", install_headers])
    compute_u_ideal = compute_u_ideal_mod.get_function('compute_u') 

    # Set up GPUArrays for intermediary data:
    ts_gpu = gpuarray.empty(N, stype)
    tsh_gpu = gpuarray.empty(N-1, stype)
    q_gpu = gpuarray.empty((N-1, 1), stype)
    G_gpu = gpuarray.empty((N-1, N-1), stype) 

    # Load data into device memory:
    s_gpu = gpuarray.to_gpu(s)

    # Get required block/grid sizes for constructing ts, tsh, and q:
    block_dim_s, grid_dim_s = cumisc.select_block_grid_sizes(dev,
                              s_gpu.shape, max_threads_per_block)

    # Get required block/grid sizes for constructing G:
    block_dim_G, grid_dim_G = cumisc.select_block_grid_sizes(dev,
                              G_gpu.shape, max_threads_per_block)
    
    # Run the kernels:
    compute_q_ideal(s_gpu, q_gpu,
                    stype(b), stype(d), stype(C), np.uint32(N-1),
                    block=block_dim_s, grid=grid_dim_s)
    compute_ts_ideal(s_gpu, ts_gpu, np.uint32(N),
                     block=block_dim_s, grid=grid_dim_s)
    compute_tsh_ideal(ts_gpu, tsh_gpu, np.uint32(N-1),
                      block=block_dim_s, grid=grid_dim_s)
    compute_G_ideal(ts_gpu, tsh_gpu, G_gpu,
                    stype(bw), np.uint32((N-1)**2),
                    block=block_dim_G, grid=grid_dim_G)

    # Free ts:
    ts_gpu.gpudata.free()
    
    # Compute pseudoinverse of G:
    G_inv_gpu = culinalg.pinv(G_gpu, dev, __pinv_rcond__)    
    
    # Compute the reconstruction coefficients:
    c_gpu = culinalg.dot(G_inv_gpu, q_gpu)
    
    # Free G, G_inv and q:
    G_gpu.gpudata.free()
    del G_gpu
    G_inv_gpu.gpudata.free()
    del G_inv_gpu
    q_gpu.gpudata.free()
    del q_gpu

    # Allocate arrays needed for reconstruction; notice that dur+dt is
    # used as the upper bound for t because gpuarray.arange() seems to
    # behave slightly differently than np.arange():
    # XXX: t_gpu could be dispensed with if the kernel were to compute
    # the times explicitly:
    t_gpu = gpuarray.arange(0, dur+dt, dt, dtype=stype) 
    u_rec_gpu = gpuarray.zeros_like(t_gpu)

    # Get required block/grid sizes for constructing u:
    block_dim_t, grid_dim_t = cumisc.select_block_grid_sizes(dev,
                              t_gpu.shape, max_threads_per_block)

    # Reconstruct signal:
    compute_u_ideal(u_rec_gpu, t_gpu, tsh_gpu,
                    c_gpu, stype(bw),
                    np.uint32(t_gpu.size), np.uint32(N-1),
                    block=block_dim_t, grid=grid_dim_t)
    u_rec = u_rec_gpu.get()

    return u_rec

