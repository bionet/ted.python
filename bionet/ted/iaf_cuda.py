#!/usr/bin/env python

"""
Time encoding and decoding algorithms that make use of the
integrate-and-fire neuron model.

These functions make use of CUDA.

- iaf_decode            - IAF time decoding machine.
- iaf_encode            - IAF time encoding machine.

"""

__all__ = ['iaf_encode', 'iaf_decode']

from string import Template
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np
from numpy import ceil, inf
from scipy.signal import resample

import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc

# Get installation location of C headers:
from scikits.cuda import install_headers

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

// u: input signal
// s: returned array of spike times
// ns: initial length of spike train
// dt: time resolution
// b: neuron biases
// d: neuron thresholds
// R: neuron resistances
// C: neuron capacitances
// y: initial value of integrator
// interval: initial value of the neuron time interval
// use_trapz: use trapezoidal integration if set to 1
// Nu: length of u:
__global__ void iaf_encode(FLOAT *u, FLOAT *s,
                           unsigned int *ns, FLOAT dt,
                           FLOAT b, FLOAT d,
                           FLOAT R, FLOAT C,
                           FLOAT *y, FLOAT *interval,
                           unsigned int use_trapz,
                           unsigned int Nu)
{
    unsigned int idx = threadIdx.x;

    FLOAT y_curr, interval_curr;
    unsigned int ns_curr, last;
    FLOAT RC = R*C;
    
    if (idx == 0) {
        y_curr = y[0];
        interval_curr = interval[0];
        ns_curr = ns[0];

        // Use the exponential Euler method when the neuron resistance
        // is not infinite:
        if ((use_trapz == 1) && isinf(R))
            last = Nu-1;
        else
            last = Nu;

        for (unsigned int i = 0; i < last; i++) {
            if isinf(R)
                if (use_trapz == 1)
                    y_curr += dt*(b+(u[i]+u[i+1])/2)/C;
                else
                    y_curr += dt*(b+u[i])/C;
            else
                y_curr = y_curr*EXP(-dt/RC)+R*(1.0-EXP(-dt/RC))*(b+u[i]);
            interval_curr += dt;
            if (y_curr >= d) {
                s[ns_curr] = interval_curr;
                interval_curr = 0;
                y_curr -= d;
                ns_curr++;
            }
        }

        // Save the integrator and interval values for the next
        // iteration:
        y[0] = y_curr;
        interval[0] = interval_curr;
        ns[0] = ns_curr;
    }                     
}
""")

def iaf_encode(u, dt, b, d, R=inf, C=1.0, dte=0.0, y=0.0, interval=0.0,
               quad_method='trapz', full_output=False):
    """
    IAF time encoding machine.
    
    Encode a finite length signal with an Integrate-and-Fire neuron.

    Parameters
    ----------
    u : array_like of floats
        Signal to encode.
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is 1/dt Hz.
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.
    dte : float
        Sampling resolution assumed by the encoder (s).
        This may not exceed `dt`.
    y : float
        Initial value of integrator.
    interval : float
        Time since last spike (in s).
    quad_method : {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal) when the
        neuron is ideal; exponential Euler integration is used
        when the neuron is leaky.
    full_output : bool
        If set, the function returns the encoded data block followed
        by the given parameters (with updated values for `y` and `interval`).
        This is useful when the function is called repeatedly to
        encode a long signal.

    Returns
    -------
    s : ndarray of floats
        If `full_output` is false, returns the signal encoded as an
        array of interspike intervals.
    [s, dt, b, d, R, C, dte, y, interval, quad_method, full_output] : list
        If `full_output` is true, returns the encoded signal
        followed by updated encoder parameters.
        
    Notes
    -----
    When trapezoidal integration is used, the value of the integral
    will not be computed for the very last entry in `u`.

    """

    # Input sanity check:
    float_type = u.dtype.type
    if float_type == np.float32:
        use_double = 0
    elif float_type == np.float64:
        use_double = 1
    else:
        raise ValueError('unsupported data type')

    # Handle empty input:
    Nu = len(u)
    if Nu == 0:
        if full_output:
            return array((),float), dt, b, d, R, C, dte, y, interval, \
                   quad_method, full_output
        else:
            return array((),float)
    
    # Check whether the encoding resolution is finer than that of the
    # original sampled signal:
    if dte > dt:
        raise ValueError('encoding time resolution must not exceeed original signal resolution')
    if dte < 0:
        raise ValueError('encoding time resolution must be nonnegative')
    if dte != 0 and dte != dt:

        # Resample signal and adjust signal length accordingly:
        M = int(dt/dte)
        u = resample(u, len(u)*M)
        Nu *= M
        dt = dte

    dev = cumisc.get_current_device()
    
    # Configure kernel:
    iaf_encode_mod = \
                   SourceModule(iaf_encode_mod_template.substitute(use_double=use_double)) 
    iaf_encode = iaf_encode_mod.get_function("iaf_encode")

    # XXX: A very long s array might cause memory problems:
    s = np.zeros(Nu, float_type)
    i_s_0 = np.zeros(1, np.uint32)
    y_0 = np.asarray([y], float_type)
    interval_0 = np.asarray([interval], float_type)
    iaf_encode(drv.In(u), drv.Out(s), drv.InOut(i_s_0),
               float_type(dt), float_type(b),
               float_type(d), float_type(R), float_type(C), 
               drv.InOut(y_0), drv.InOut(interval_0),
               np.uint32(True if quad_method == 'trapz' else False),
               np.uint32(Nu),
               block=(1, 1, 1))

    if full_output:
        return s[0:i_s_0[0]], dt, b, d, R, C, y_0[0], interval_0[0], \
               quad_method, full_output
    else:
        return s[0:i_s_0[0]]

# Kernel template for computing q for the ideal IAF time decoder:
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

# Kernel template for computing spike times for the ideal IAF time decoder:
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

# Kernel template for computing midpoints between spikes for the ideal
# IAF time decoder:
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

# Kernel template for computing the recovery matrix for the ideal IAF
# time decoder:
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

# Kernel template for reconstructing the encoded signal for the ideal
# IAF time decoder:
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
__global__ void compute_u(FLOAT *u_rec, FLOAT *tsh, FLOAT *c,
                          FLOAT bw, FLOAT dt, unsigned Nt, unsigned int Nsh) {
    unsigned int idx = blockIdx.y*${max_threads_per_block}*${max_blocks_per_grid}+
                       blockIdx.x*${max_threads_per_block}+threadIdx.x;
    FLOAT bwpi = bw/PI;
    FLOAT u_temp = 0;
    
    // Each thread reconstructs the signal at time t[idx]:
    if (idx < Nt) {
        for (unsigned int i = 0; i < Nsh; i++) {
            u_temp += SINC(bwpi*(idx*dt-tsh[i]))*bwpi*c[i];
        }
        u_rec[idx] = u_temp;
    }
}
""")

def iaf_decode(s, dur, dt, bw, b, d, R=inf, C=1.0):
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
        
    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.

    """

    # Input sanity check:
    float_type = s.dtype.type
    if float_type == np.float32:
        use_double = 0
    elif float_type == np.float64:
        use_double = 1
    else:
        raise ValueError('unsupported data type')        

    N = len(s)
    
    if not np.isinf(R):
        raise ValueError('decoding for leaky neuron not implemented yet')

    dev = cumisc.get_current_device()
                                    
    # Get device constraints:
    max_threads_per_block, max_block_dim, max_grid_dim = cumisc.get_dev_attrs(dev)
    max_blocks_per_grid = max(max_grid_dim)

    # Use a smaller block size than the maximum to prevent the kernels
    # from using too many registers:
    max_threads_per_block = 256
    
    # Prepare kernels:
    compute_q_ideal_mod = \
                        SourceModule(compute_q_ideal_mod_template.substitute(use_double=use_double,
                                     max_threads_per_block=max_threads_per_block,
                                     max_blocks_per_grid=max_blocks_per_grid))
    compute_q_ideal = \
                    compute_q_ideal_mod.get_function('compute_q')

    compute_ts_ideal_mod = \
                         SourceModule(compute_ts_ideal_mod_template.substitute(use_double=use_double,
                                     max_threads_per_block=max_threads_per_block,
                                     max_blocks_per_grid=max_blocks_per_grid))    
    compute_ts_ideal = \
                     compute_ts_ideal_mod.get_function('compute_ts')

    compute_tsh_ideal_mod = \
                          SourceModule(compute_tsh_ideal_mod_template.substitute(use_double=use_double,
                                     max_threads_per_block=max_threads_per_block,
                                     max_blocks_per_grid=max_blocks_per_grid)) 
    compute_tsh_ideal = \
                      compute_tsh_ideal_mod.get_function('compute_tsh')
                          

    compute_G_ideal_mod = \
                        SourceModule(compute_G_ideal_mod_template.substitute(use_double=use_double,
                                     max_threads_per_block=max_threads_per_block,
                                     max_blocks_per_grid=max_blocks_per_grid,
                                     cols=(N-1)),
                                     options=['-I', install_headers])
    compute_G_ideal = compute_G_ideal_mod.get_function('compute_G') 

    compute_u_ideal_mod = \
                        SourceModule(compute_u_ideal_mod_template.substitute(use_double=use_double,
                                     max_threads_per_block=max_threads_per_block,
                                     max_blocks_per_grid=max_blocks_per_grid),
                                     options=["-I", install_headers])
    compute_u_ideal = compute_u_ideal_mod.get_function('compute_u') 

    # Load data into device memory:
    s_gpu = gpuarray.to_gpu(s)

    # Set up GPUArrays for intermediary data:
    ts_gpu = gpuarray.empty(N, float_type)
    tsh_gpu = gpuarray.empty(N-1, float_type)
    q_gpu = gpuarray.empty((N-1, 1), float_type)
    G_gpu = gpuarray.empty((N-1, N-1), float_type) 

    # Get required block/grid sizes for constructing ts, tsh, and q:
    block_dim_s, grid_dim_s = cumisc.select_block_grid_sizes(dev,
                              s_gpu.shape, max_threads_per_block)

    # Get required block/grid sizes for constructing G:
    block_dim_G, grid_dim_G = cumisc.select_block_grid_sizes(dev,
                              G_gpu.shape, max_threads_per_block)
    
    # Run the kernels:
    compute_q_ideal(s_gpu, q_gpu,
                    float_type(b), float_type(d), float_type(C), np.uint32(N-1),
                    block=block_dim_s, grid=grid_dim_s)
    compute_ts_ideal(s_gpu, ts_gpu, np.uint32(N),
                     block=block_dim_s, grid=grid_dim_s)
    compute_tsh_ideal(ts_gpu, tsh_gpu, np.uint32(N-1),
                      block=block_dim_s, grid=grid_dim_s)
    compute_G_ideal(ts_gpu, tsh_gpu, G_gpu,
                    float_type(bw), np.uint32((N-1)**2),
                    block=block_dim_G, grid=grid_dim_G)

    # Free unneeded s and ts to provide more memory to the pinv computation:
    del s_gpu, ts_gpu
    
    # Compute pseudoinverse of G:
    G_inv_gpu = culinalg.pinv(G_gpu, __pinv_rcond__)    
    
    # Compute the reconstruction coefficients:
    c_gpu = culinalg.dot(G_inv_gpu, q_gpu)
    
    # Free unneeded G, G_inv and q:
    del G_gpu, G_inv_gpu, q_gpu

    # Allocate array for reconstructed signal:
    Nt = int(ceil(dur/dt))
    u_rec_gpu = gpuarray.zeros(Nt, float_type)

    # Get required block/grid sizes for constructing u:
    block_dim_t, grid_dim_t = cumisc.select_block_grid_sizes(dev,
                              Nt, max_threads_per_block)

    # Reconstruct signal:
    compute_u_ideal(u_rec_gpu, tsh_gpu,
                    c_gpu, float_type(bw), float_type(dt),
                    np.uint32(Nt), np.uint32(N-1),
                    block=block_dim_t, grid=grid_dim_t)
    u_rec = u_rec_gpu.get()

    return u_rec

