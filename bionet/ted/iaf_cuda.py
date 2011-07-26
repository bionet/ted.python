#!/usr/bin/env python

"""
Time encoding and decoding algorithms that make use of the
integrate-and-fire neuron model.

- iaf_decode            - IAF time decoding machine.
- iaf_decode_pop        - MISO IAF time decoding machine.
- iaf_encode            - IAF time encoding machine.
- iaf_encode_pop        - SIMO IAF time encoding machine.

These functions make use of CUDA.

"""

__all__ = ['iaf_decode', 'iaf_decode_pop', 'iaf_encode', 'iaf_encode_pop']

from string import Template
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np
from scipy.signal import resample

import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc

# Get installation location of C headers:
from scikits.cuda import install_headers

# Kernel template for performing ideal/leaky IAF time encoding using a
# single encoder:
iaf_encode_template = Template("""
#if ${use_double}
#define FLOAT double
#else
#define FLOAT float
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
                y_curr = y_curr*exp(-dt/RC)+R*(1.0-exp(-dt/RC))*(b+u[i]);
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

def iaf_encode(u, dt, b, d, R=np.inf, C=1.0, dte=0.0, y=0.0, interval=0.0,
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
                   SourceModule(iaf_encode_template.substitute(use_double=use_double)) 
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

compute_q_template = Template("""
#include <pycuda/pycuda-complex.hpp>

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#endif

// N must equal one less the length of s:
__global__ void compute_q_ideal(FLOAT *s, COMPLEX *q, FLOAT b,
                                FLOAT d, FLOAT C, unsigned int N) {                          
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N) {
        q[idx] = COMPLEX(C*d-b*s[idx+1]);
    }
}

__global__ void compute_q_leaky(FLOAT *s, COMPLEX *q, FLOAT b,
                                FLOAT d, FLOAT R, FLOAT C, unsigned int N) {                          
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    FLOAT RC = R*C;
    if (idx < N) {
        q[idx] = COMPLEX(C*(d+b*R*(exp(-s[idx+1]/RC)-1.0)));
    }
}
""")

compute_ts_template = Template("""
#if ${use_double}
#define FLOAT double
#else
#define FLOAT float
#endif

// N == len(s)
__global__ void compute_ts(FLOAT *s, FLOAT *ts, unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N) {
        ts[idx] = 0.0;
        for (unsigned int i = 0; i < idx+1; i++)
            ts[idx] += s[i]; 
    }
}
""")

compute_tsh_template = Template("""
#if ${use_double}
#define FLOAT double
#else
#define FLOAT float
#endif

// Nsh == len(ts)-1
__global__ void compute_tsh(FLOAT *ts, FLOAT *tsh, unsigned int Nsh) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < Nsh) {
        tsh[idx] = (ts[idx]+ts[idx+1])/2;
    }
}   
""")

compute_G_template = Template("""
#include <pycuda/pycuda-complex.hpp>

#include <cuConstants.h>    // needed to provide PI
#include <cuSpecialFuncs.h> // needed to provide sici()

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#define SICI(x, si, ci) sici(x, si, ci)
#define EXPI(z) expi(z)
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#define SICI(x, si, ci) sicif(x, si, ci)
#define EXPI(z) expif(z)
#endif

// N must equal the square of one less than the length of ts:
__global__ void compute_G_ideal(FLOAT *ts, FLOAT *tsh, COMPLEX *G,
                                FLOAT bw, unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int ix = idx/${cols};
    unsigned int iy = idx%${cols};
    FLOAT si0, si1, ci;
    
    if (idx < N) {
        SICI(bw*(ts[ix+1]-tsh[iy]), &si1, &ci);
        SICI(bw*(ts[ix]-tsh[iy]), &si0, &ci);
        G[idx] = COMPLEX((si1-si0)/PI);
    }
}

__global__ void compute_G_leaky(FLOAT *ts, FLOAT *tsh, COMPLEX *G,
                                FLOAT bw, FLOAT R, FLOAT C,
                                unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int ix = idx/${cols};
    unsigned int iy = idx%${cols};
    FLOAT RC = R*C;
    
    if (idx < N) {
        if ((ts[ix] < tsh[iy]) && (tsh[iy] < ts[ix+1])) {
            G[idx] = COMPLEX(0,-1.0/4.0)*exp((tsh[iy]-ts[ix+1])/RC)*
                (FLOAT(2.0)*EXPI(COMPLEX(1,-RC*bw)*(ts[ix]-tsh[iy])/RC)-
                 FLOAT(2.0)*EXPI(COMPLEX(1,-RC*bw)*(ts[ix+1]-tsh[iy])/RC)-
                 FLOAT(2.0)*EXPI(COMPLEX(1,RC*bw)*(ts[ix]-tsh[iy])/RC)+
                 FLOAT(2.0)*EXPI(COMPLEX(1,RC*bw)*(ts[ix+1]-tsh[iy])/RC)+
                 log(COMPLEX(-1,-RC*bw))+log(COMPLEX(1,-RC*bw))-
                 log(COMPLEX(-1,RC*bw))-log(COMPLEX(1,RC*bw))+
                 log(COMPLEX(0,-1)/COMPLEX(RC*bw,-1))-log(COMPLEX(0,1)/COMPLEX(RC*bw,-1))+
                 log(COMPLEX(0,-1)/COMPLEX(RC*bw,1))-log(COMPLEX(0,1)/COMPLEX(RC*bw,1)))/FLOAT(PI);
        } else {
            G[idx] = COMPLEX(0,-1.0/2.0)*exp((tsh[iy]-ts[ix+1])/RC)* 
                (EXPI(COMPLEX(1,-RC*bw)*(ts[ix]-tsh[iy])/RC)-
                 EXPI(COMPLEX(1,-RC*bw)*(ts[ix+1]-tsh[iy])/RC)-
                 EXPI(COMPLEX(1,RC*bw)*(ts[ix]-tsh[iy])/RC)+
                 EXPI(COMPLEX(1,RC*bw)*(ts[ix+1]-tsh[iy])/RC))/FLOAT(PI);
        }
    }
}
""")

compute_u_template = Template("""
#include <cuConstants.h>    // needed to provide PI
#include <cuSpecialFuncs.h> // needed to provide sinc()

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#define SINC(x) sinc(x)
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#define SINC(x) sincf(x)
#endif

// Nt == len(t)
// Nsh == len(tsh)
__global__ void compute_u(COMPLEX *u_rec, COMPLEX *c, FLOAT *tsh, 
                          FLOAT bw, FLOAT dt, unsigned Nt, unsigned int Nsh) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    FLOAT bwpi = bw/PI;
    
    // Each thread reconstructs the signal at time t[idx]:
    if (idx < Nt) {
        COMPLEX u_temp = COMPLEX(0);
        for (unsigned int i = 0; i < Nsh; i++) {
            u_temp += SINC(bwpi*(idx*dt-tsh[i]))*bwpi*c[i];
        }
        u_rec[idx] = u_temp;
    }
}
""")

def iaf_decode(s, dur, dt, bw, b, d, R=np.inf, C=1.0):
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

    N = len(s)
    float_type = s.dtype.type
    if float_type == np.float32:
        use_double = 0
        complex_type = np.complex64
        __pinv_rcond__ = 1e-4
    elif float_type == np.float64:
        use_double = 1
        complex_type = np.complex128
        __pinv_rcond__ = 1e-8
    else:
        raise ValueError('unsupported data type')        
        
    # Prepare kernels:
    compute_ts_mod = \
                   SourceModule(compute_ts_template.substitute(use_double=use_double))
    compute_ts = \
               compute_ts_mod.get_function('compute_ts')

    compute_tsh_mod = \
                    SourceModule(compute_tsh_template.substitute(use_double=use_double))
    compute_tsh = \
                compute_tsh_mod.get_function('compute_tsh')

    compute_q_mod = \
                  SourceModule(compute_q_template.substitute(use_double=use_double))
    compute_q_ideal = \
                    compute_q_mod.get_function('compute_q_ideal')
    compute_q_leaky = \
                    compute_q_mod.get_function('compute_q_leaky')
                          
    compute_G_mod = \
                  SourceModule(compute_G_template.substitute(use_double=use_double,
                                                             cols=(N-1)),
                               options=['-I', install_headers])
    compute_G_ideal = compute_G_mod.get_function('compute_G_ideal') 
    compute_G_leaky = compute_G_mod.get_function('compute_G_leaky') 

    compute_u_mod = \
                  SourceModule(compute_u_template.substitute(use_double=use_double),
                               options=["-I", install_headers])
    compute_u = compute_u_mod.get_function('compute_u') 
    
    # Load data into device memory:
    s_gpu = gpuarray.to_gpu(s)

    # Set up GPUArrays for intermediary data:
    ts_gpu = gpuarray.empty(N, float_type)
    tsh_gpu = gpuarray.empty(N-1, float_type)
    q_gpu = gpuarray.empty((N-1, 1), complex_type)
    G_gpu = gpuarray.empty((N-1, N-1), complex_type) 

    # Get required block/grid sizes for constructing ts, tsh, and q;
    # use a smaller block size than the maximum to prevent the kernels
    # from using too many registers:
    dev = cumisc.get_current_device()
    max_threads_per_block = 128
    block_dim_s, grid_dim_s = \
                 cumisc.select_block_grid_sizes(dev, s_gpu.shape, max_threads_per_block)
    
    # Get required block/grid sizes for constructing G:
    block_dim_G, grid_dim_G = \
                 cumisc.select_block_grid_sizes(dev, G_gpu.shape, max_threads_per_block)
    
    # Run the kernels:
    compute_ts(s_gpu, ts_gpu, np.uint32(N),
               block=block_dim_s, grid=grid_dim_s)
    compute_tsh(ts_gpu, tsh_gpu, np.uint32(N-1),
                block=block_dim_s, grid=grid_dim_s)
    if np.isinf(R):
        compute_q_ideal(s_gpu, q_gpu,
                        float_type(b), float_type(d), float_type(C), np.uint32(N-1),
                        block=block_dim_s, grid=grid_dim_s)
        compute_G_ideal(ts_gpu, tsh_gpu, G_gpu,
                        float_type(bw), np.uint32((N-1)**2),
                        block=block_dim_G, grid=grid_dim_G)
    else:
        compute_q_leaky(s_gpu, q_gpu,
                        float_type(b), float_type(d),
                        float_type(R), float_type(C), np.uint32(N-1),
                        block=block_dim_s, grid=grid_dim_s)
        compute_G_leaky(ts_gpu, tsh_gpu, G_gpu,
                        float_type(bw), float_type(R), float_type(C),
                        np.uint32((N-1)**2),
                        block=block_dim_G, grid=grid_dim_G)
    
    # Free unneeded s and ts to provide more memory to the pinv computation:
    del s_gpu, ts_gpu
    
    # Compute the reconstruction coefficients:
    c_gpu = culinalg.dot(culinalg.pinv(G_gpu, __pinv_rcond__), q_gpu)
    
    # Free unneeded G, G_inv and q:
    del G_gpu, q_gpu

    # Allocate array for reconstructed signal:
    Nt = int(np.ceil(dur/dt))

    u_rec_gpu = gpuarray.to_gpu(np.zeros(Nt, complex_type))
    ### Replace the above with the following line when the bug in
    # gpuarray.zeros in pycuda 2011.1.2 is fixed:
    #u_rec_gpu = gpuarray.zeros(Nt, complex_type)

    # Get required block/grid sizes for constructing u:
    block_dim_t, grid_dim_t = \
                 cumisc.select_block_grid_sizes(dev, Nt, max_threads_per_block)
                              
    # Reconstruct signal:
    compute_u(u_rec_gpu, c_gpu,
              tsh_gpu, float_type(bw), float_type(dt),                    
              np.uint32(Nt), np.uint32(N-1),
              block=block_dim_t, grid=grid_dim_t)
    u_rec = u_rec_gpu.get()

    return np.real(u_rec)

# Kernel template for performing ideal/leaky time encoding a
# 1D signal using N encoders:
iaf_encode_pop_template = Template("""
#if ${use_double}
#define FLOAT double
#else
#define FLOAT float
#endif

// Macro for accessing multidimensional arrays with cols columns by
// linear index:
#define INDEX(row, col, cols) (row*cols+col)

// u: input signal
// s: returned matrix of spike trains
// ns: returned lengths of spike trains
// dt: time resolution
// b: neuron biases
// d: neuron thresholds
// R: neuron resistances
// C: neuron capacitances
// y: initial values of integrators
// interval: initial values of the neuron time intervals
// use_trapz: use trapezoidal integration if set to 1
// Nu: length of u
// N: length of ns, b, d, R, C, y, and interval:
__global__ void iaf_encode_pop(FLOAT *u, FLOAT *s,
                           unsigned int *ns, FLOAT dt,
                           FLOAT *b, FLOAT *d,
                           FLOAT *R, FLOAT *C,
                           FLOAT *y, FLOAT *interval,
                           unsigned int use_trapz,
                           unsigned int Nu,
                           unsigned int N)
{
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    FLOAT y_curr, interval_curr;                       
    FLOAT u_curr, u_next, b_curr, d_curr, R_curr, C_curr, RC_curr;
    unsigned int ns_curr, last;
    
    if (idx < N) {
        // Initialize integrator accumulator, interspike interval,
        // and the spike counter for the current train:
        y_curr = y[idx];
        interval_curr = interval[idx];
        ns_curr = ns[idx]; 

        b_curr = b[idx];
        d_curr = d[idx];
        R_curr = R[idx];
        C_curr = C[idx];
        RC_curr = R_curr*C_curr;

        // Use the exponential Euler method when the neuron resistance
        // is not infinite:
        if ((use_trapz == 1) && isinf(R_curr))
            last = Nu-1;
        else
            last = Nu;

        for (unsigned int i = 0; i < last; i++) {
            u_curr = u[i];
            u_next = u[i+1];
            if isinf(R_curr) {
                if (use_trapz == 1)
                    y_curr += dt*(b_curr+(u_curr+u_next)/2)/C_curr;
                else
                    y_curr += dt*(b_curr+u_curr)/C_curr;
            } else 
                y_curr = y_curr*exp(-dt/RC_curr)+
                         R_curr*(1-exp(-dt/RC_curr))*(b_curr+u_curr);                        
            
            interval_curr += dt;
            if (y_curr >= d_curr) {
                s[INDEX(idx, ns_curr, Nu)] = interval_curr;
                interval_curr = 0.0;
                y_curr -= d_curr;
                ns_curr++;
            }
        }

        // Save the integrator and interval values for the next
        // iteration:
        y[idx] = y_curr;
        interval[idx] = interval_curr;
        ns[idx] = ns_curr;
    }                     
}
""")

def iaf_encode_pop(u_gpu, dt, b_gpu, d_gpu, R_gpu, C_gpu,
                   y_gpu=None, interval_gpu=None,
                   quad_method='trapz', full_output=False):
    """
    Population IAF time encoding machine.

    Encode a finite length signal with a population of Integrate-and-Fire
    Neurons.
    
    Parameters
    ----------
    u_gpu : pycuda.gpuarray.GPUArray
        Signal to encode.
    dt : float
        Sampling resolution of input signal; the sampling frequency is
        1/dt Hz.
    b_gpu : pycuda.gpuarray.GPUArray
        Array of encoder biases.
    d_gpu : pycuda.gpuarray.GPUArray
        Array of encoder thresholds.
    R_gpu : pycuda.gpuarray.GPUArray
        Array of neuron resistances.
    C_gpu : pycuda.gpuarray.GPUArray
        Array of neuron capacitances.
    y_gpu : pycuda.gpuarray.GPUArray
        Initial values of integrators.
    interval_gpu : pycuda.gpuarray.GPUArray
        Times since last spike (in s) for each neuron.
    quad_method : {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal) when the
        neuron is ideal; exponential Euler integration is used
        when the neuron is leaky.
    full_output : bool
        If true, the function returns the updated arrays `y_gpu` and
        `interval_gpu` in addition to the the encoded data block.

    Returns
    -------
    [s_gpu, ns_gpu] : list of pycuda.gpuarray.GPUArray
        If `full_output` is false, returns the encoded signal as a
        matrix `s_gpu` whose rows contain the spike times generated by each
        neuron. The number of spike times in each row is returned in
        `ns_gpu`; all other values in `s_gpu` are set to 0.
    [s_gpu, ns_gpu, y_gpu, interval_gpu] : list of pycuda.gpuarray.GPUArray
        If `full_output` is true, returns the encoded signal
        followed by updated encoder parameters.

    """

    float_type = u_gpu.dtype.type
    if float_type == np.float32:
        use_double = 0
    elif float_type == np.float64:
        use_double = 1
    else:
        raise ValueError('unsupported data type')

    # Get the length of the signal:
    Nu = u_gpu.size

    N = b_gpu.size
    if (d_gpu.size != N) or \
           (R_gpu.size != N) or (C_gpu.size != N):
        raise ValueError('parameter arrays must be of same length')
    if ((y_gpu != None) and (y_gpu.size != N)) or \
       ((interval_gpu != None) and (interval_gpu.size != N)):
        raise ValueError('parameter arrays must be of same length')

    dev = cumisc.get_current_device()
    
    # Use a smaller block size than the maximum to prevent the kernels
    # from using too many registers:
    max_threads_per_block = 256

    # Get required block/grid sizes for running N encoders to process
    # the N signals:
    block_dim, grid_dim = cumisc.select_block_grid_sizes(dev, N,
                          max_threads_per_block)

    # Configure kernel:
    cache_dir = None
    iaf_encode_pop_mod = \
                   SourceModule(iaf_encode_pop_template.substitute(use_double=use_double),
                                cache_dir=cache_dir)
    iaf_encode_pop = iaf_encode_pop_mod.get_function("iaf_encode_pop")

    # Initialize integrator variables if necessary:
    if y_gpu == None:
        y_gpu = gpuarray.zeros(N, float_type)
    if interval_gpu == None:
        interval_gpu = gpuarray.zeros(N, float_type)

    # XXX: A very long s array might cause memory problems:
    s_gpu = gpuarray.zeros((N, Nu), float_type)
    ns_gpu = gpuarray.zeros(N, np.uint32)
    iaf_encode_pop(u_gpu, s_gpu, ns_gpu,
                   float_type(dt), b_gpu, d_gpu,
                   R_gpu, C_gpu,                   
                   y_gpu, interval_gpu,
                   np.uint32(True if quad_method == 'trapz' else False),
                   np.uint32(Nu),
                   np.uint32(N),
                   block=block_dim, grid=grid_dim)

    if full_output:
        return [s_gpu, ns_gpu, y_gpu, interval_gpu]
    else:
        return [s_gpu, ns_gpu]

def _compute_idx_map(ns):
    """
    Map a linear index to corresponding neuron and interspike interval indices.

    Parameters
    ----------
    ns : ndarray
        `ns[i]` is the number of interspike intervals produced by
         neuron `i`.

    Returns
    -------
    idx_to_ni : ndarray
        Map of linear index to neuron index.
    idx_to_k : ndarray
        Map of linear index to interspike interval index.
        
    Notes
    -----
    The relationship between the linear index and the output arrays is
    as follows:
    
    idx | idx_to_ni | idx_to_k
    ----+-----------+---------
     0  |     0     |    0
     1  |     0     |    1 
     2  |     1     |    0
     3  |     1     |    1
     4  |     1     |    2
     5  |     2     |    0
     6  |     2     |    1

    The number of interspike intervals per neuron is decremented by
    one for each neuron that has generated more than 1 spike.

    This function should be reimplemented to run directly on the GPU.
    
    """

    # Number of neurons:
    N = len(ns)

    # Number of index values:
    Nidx = np.sum(ns)-np.sum(ns>1)
    
    # Map from index to neuron index:
    idx_to_ni = np.empty(Nidx, np.uint32)

    # Map from index to interspike interval index:
    idx_to_k = np.empty(Nidx, np.uint32)

    idx = 0
    for ni in xrange(N):
        for k in xrange(ns[ni]-1):
            idx_to_ni[idx] = ni
            idx_to_k[idx] = k
            idx += 1

    return idx_to_ni, idx_to_k

compute_q_pop_template = Template("""
#include <pycuda/pycuda-complex.hpp>

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#define EXP(x) exp(x)
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#define EXP(x) expf(x)
#endif

#define INDEX(row,col,cols) row*cols+col

__global__ void compute_q_ideal(FLOAT *s, COMPLEX *q, FLOAT *b, 
                                FLOAT *d, FLOAT *C,
                                unsigned int *idx_to_ni,
                                unsigned int *idx_to_k,
                                unsigned int s_cols,                          
                                unsigned int Nq) {                          
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < Nq) {
        unsigned int ni = idx_to_ni[idx];
        unsigned int k = idx_to_k[idx];
        
        q[idx] = C[ni]*d[ni]-b[ni]*s[INDEX(ni,k+1,s_cols)];
    }
}

__global__ void compute_q_leaky(FLOAT *s, COMPLEX *q, FLOAT *b, 
                                FLOAT *d, FLOAT *R, FLOAT *C,
                                unsigned int *idx_to_ni,
                                unsigned int *idx_to_k,
                                unsigned int s_cols,                          
                                unsigned int Nq) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < Nq) {
        unsigned int ni = idx_to_ni[idx];
        unsigned int k = idx_to_k[idx];
        
        q[idx] = C[ni]*(d[ni]+b[ni]*R[ni]*(EXP(-s[INDEX(ni,k+1,s_cols)]/(R[ni]*C[ni]))-1));
    }
}                                
""")

compute_ts_pop_template = Template("""
#if ${use_double}
#define FLOAT double
#else
#define FLOAT float
#endif

#define INDEX(row,col,cols) row*cols+col

// N: number of rows in s
__global__ void compute_ts(FLOAT *s, unsigned int *ns,
                           FLOAT *ts,
                           unsigned int s_cols,
                           unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N) {
        FLOAT temp = 0.0;
        unsigned int j;
        for (unsigned int i = 0; i < ns[idx]; i++) {
           j = INDEX(idx,i,s_cols);
           temp += s[j];
           ts[j] = temp;
        }
    }
}
""")

compute_tsh_pop_template = Template("""
#if ${use_double}
#define FLOAT double
#else
#define FLOAT float
#endif

#define INDEX(row,col,cols) row*cols+col

// N: number of rows
__global__ void compute_tsh(FLOAT *ts, unsigned int *ns,
                            FLOAT *tsh,
                            unsigned int s_cols,
                            unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N) {
        FLOAT temp = 0.0;
        unsigned int j_curr, j_next;
        for (unsigned int i = 0; i < ns[idx]-1; i++) {
            j_curr = INDEX(idx,i,s_cols);
            j_next = INDEX(idx,i+1,s_cols);
            tsh[j_curr] = (ts[j_curr]+ts[j_next])/2;
        }
    }
}                            
""")

compute_G_pop_template = Template("""
#include <cuConstants.h>    // needed to provide PI
#include <cuSpecialFuncs.h> // needed to provide sici()

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#define SICI(x, si, ci) sici(x, si, ci)
#define EXP(x) exp(x)
#define EXPI(z) expi(z)
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#define SICI(x, si, ci) sicif(x, si, ci)
#define EXP(x) expf(x)
#define EXPI(z) expif(z)
#endif

#define INDEX(row,col,cols) row*cols+col

// N: total number of entries in G
__global__ void compute_G_ideal(FLOAT *ts, FLOAT *tsh, COMPLEX *G, FLOAT bw,
                                unsigned int *idx_to_ni,
                                unsigned int *idx_to_k,
                                unsigned int Nq,
                                unsigned int s_cols,                              
                                unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int row = idx/Nq;
    unsigned int col = idx%Nq;
    FLOAT si0, si1, ci;

    if (idx < N) {

        unsigned int l = idx_to_ni[row];
        unsigned int m = idx_to_ni[col];
        unsigned int n = idx_to_k[row];
        unsigned int k = idx_to_k[col];
        
        SICI(bw*(ts[INDEX(l,n,s_cols)]-tsh[INDEX(m,k,s_cols)]), &si1, &ci);
        SICI(bw*(ts[INDEX(l,n+1,s_cols)]-tsh[INDEX(m,k,s_cols)]), &si0, &ci);
        G[idx] = COMPLEX((si0-si1)/PI);
    }
}

__global__ void compute_G_leaky(FLOAT *ts, FLOAT *tsh, COMPLEX *G, FLOAT bw,
                                FLOAT *R, FLOAT *C,
                                unsigned int *idx_to_ni,
                                unsigned int *idx_to_k,
                                unsigned int Nq,
                                unsigned int s_cols,                              
                                unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int row = idx/Nq;
    unsigned int col = idx%Nq;

    if (idx < N) {

        unsigned int l = idx_to_ni[row];
        unsigned int m = idx_to_ni[col];
        unsigned int n = idx_to_k[row];
        unsigned int k = idx_to_k[col];

        FLOAT RC = R[l]*C[l];
        if (ts[INDEX(l,n,s_cols)] < tsh[INDEX(m,k,s_cols)] &&
            tsh[INDEX(m,k,s_cols)] < ts[INDEX(l,n+1,s_cols)]) {
            G[idx] = COMPLEX(0,-1.0/4.0)*EXP((tsh[INDEX(m,k,s_cols)]-ts[INDEX(l,n+1,s_cols)])/RC)*
                (2.0*EXPI(COMPLEX(1,-RC*bw)*(ts[INDEX(l,n,s_cols)]-tsh[INDEX(m,k,s_cols)])/RC)-
                 2.0*EXPI(COMPLEX(1,-RC*bw)*(ts[INDEX(l,n+1,s_cols)]-tsh[INDEX(m,k,s_cols)])/RC)-
                 2.0*EXPI(COMPLEX(1,RC*bw)*(ts[INDEX(l,n,s_cols)]-tsh[INDEX(m,k,s_cols)])/RC)+
                 2.0*EXPI(COMPLEX(1,RC*bw)*(ts[INDEX(l,n+1,s_cols)]-tsh[INDEX(m,k,s_cols)])/RC)+
                 log(COMPLEX(-1,-RC*bw))+log(COMPLEX(1,-RC*bw))-
                 log(COMPLEX(-1,RC*bw))-log(COMPLEX(1,RC*bw))+
                 log(COMPLEX(0,-1)/COMPLEX(RC*bw,-1))-log(COMPLEX(0,1)/COMPLEX(RC*bw,-1))+
                 log(COMPLEX(0,-1)/COMPLEX(RC*bw,1))-log(COMPLEX(0,1)/COMPLEX(RC*bw,1)))/PI;
        } else {
            G[idx] = COMPLEX(0,-1.0/2.0)*EXP((tsh[INDEX(m,k,s_cols)]-ts[INDEX(l,n+1,s_cols)])/RC)*
                 (EXPI(COMPLEX(1,-RC*bw)*(ts[INDEX(l,n,s_cols)]-tsh[INDEX(m,k,s_cols)])/RC)-
                  EXPI(COMPLEX(1,-RC*bw)*(ts[INDEX(l,n+1,s_cols)]-tsh[INDEX(m,k,s_cols)])/RC)-
                  EXPI(COMPLEX(1,RC*bw)*(ts[INDEX(l,n,s_cols)]-tsh[INDEX(m,k,s_cols)])/RC)+
                  EXPI(COMPLEX(1,RC*bw)*(ts[INDEX(l,n+1,s_cols)]-tsh[INDEX(m,k,s_cols)])/RC))/PI;
        }
    }
}
""")

compute_u_pop_template = Template("""
#include <pycuda/pycuda-complex.hpp>
#include <cuConstants.h>              // needed to provide PI
#include <cuSpecialFuncs.h>           // needed to provide sinc()

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#define SINC(x) sinc(x)
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#define SINC(x) sincf(x)
#endif

#define INDEX(row,col,cols) row*cols+col

// u_rec: reconstructed signal
// c: reconstruction coeficients
// tsh: midpoints between spike times
// bw: bandwidth (rad/s)
// dt: time resolution of reconstructed signal
// M: number of neurons
// Nt: len(t)
__global__ void compute_u(COMPLEX *u_rec, COMPLEX *c,
                          FLOAT *tsh, unsigned int *ns,
                          FLOAT bw, FLOAT dt,
                          unsigned int s_cols,
                          unsigned int M,
                          unsigned int Nt) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    FLOAT bwpi = bw/PI;
    
    // Each thread reconstructs the signal at time t[idx]:
    if (idx < Nt) {
        COMPLEX u_temp = COMPLEX(0);
        unsigned int c_ind = 0;
        for (unsigned int m = 0; m < M; m++) {
            for (unsigned int k = 0; k < ns[m]-1; k++) {
                u_temp += SINC(bwpi*(idx*dt-tsh[INDEX(m,k,s_cols)]))*bwpi*c[c_ind+k];
            }
            if (ns[m] > 1)
                c_ind += (ns[m]-1);
            else
                c_ind += ns[m];
        }
        u_rec[idx] = u_temp;
    }
}
""")

def iaf_decode_pop(s_gpu, ns_gpu, dur, dt, bw, b_gpu, d_gpu,
                   R_gpu, C_gpu):
    """
    Multiple-input single-output IAF time decoding machine.

    Decode a signal encoded with an ensemble of Integrate-and-Fire
    neurons assuming that the encoded signal is representable in terms
    of sinc kernels.

    Parameters
    ----------
    s_gpu : pycuda.gpuarray.GPUArray
        Signal encoded by an ensemble of encoders. The nonzero
        values represent the time between spikes (in s). The number of
        arrays in the list corresponds to the number of encoders in
        the ensemble.
    ns_gpu : pycuda.gpuarray.GPUArray
        Number of interspike intervals in each row of `s_gpu`.
    dur : float
        Duration of signal (in s).
    dt : float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw : float
        Signal bandwidth (in rad/s).
    b_gpu : pycuda.gpuarray.GPUArray
        Array of encoder biases.
    d_gpu : pycuda.gpuarray.GPUArray
        Array of encoder thresholds.
    R_gpu : pycuda.gpuarray.GPUArray
        Array of neuron resistances.
    C_gpu : pycuda.gpuarray.GPUArray
        Array of neuron capacitances.
    
    Returns
    -------
    u_rec : pycuda.gpuarray.GPUArray
        Recovered signal.

    Notes
    -----
    The number of spikes contributed by each neuron may differ from the
    number contributed by other neurons.
    
    """
    
    # Sanity checks:
    float_type = s_gpu.dtype.type
    if float_type == np.float32:
        use_double = 0
        complex_type = np.complex64
        __pinv_rcond__ = 1e-4
    elif float_type == np.float64:
        use_double = 1
        complex_type = np.complex128        
        __pinv_rcond__ = 1e-8
    else:
        raise ValueError('unsupported data type')

    # Number of spike trains:
    N = s_gpu.shape[0]
    if not N:
        raise ValueError('no spike data given')
    if (ns_gpu.size != N) or (b_gpu.size != N) or (d_gpu.size != N) or \
       (R_gpu.size != N) or (C_gpu.size != N):
        raise ValueError('parameter arrays must be of same length')

    # Map CUDA index to neuron index and interspike interval index:
    ns = ns_gpu.get()
    idx_to_ni, idx_to_k = _compute_idx_map(ns)
    idx_to_ni_gpu = gpuarray.to_gpu(idx_to_ni)
    idx_to_k_gpu = gpuarray.to_gpu(idx_to_k)
    
    # Get required block/grid sizes; use a smaller block size than the
    # maximum to prevent the kernels from using too many registers:
    dev = cumisc.get_current_device()
    max_threads_per_block = 128

    # Prepare kernels:
    cache_dir = None
    compute_q_pop_mod = \
        SourceModule(compute_q_pop_template.substitute(use_double=use_double),
                     cache_dir=cache_dir)
    compute_q_ideal_pop = \
                        compute_q_pop_mod.get_function('compute_q_ideal')
    compute_q_leaky_pop = \
                        compute_q_pop_mod.get_function('compute_q_leaky')

    compute_ts_pop_mod = \
        SourceModule(compute_ts_pop_template.substitute(use_double=use_double),
                     cache_dir=cache_dir)
    compute_ts_pop = \
                   compute_ts_pop_mod.get_function('compute_ts')
    
    compute_tsh_pop_mod = \
        SourceModule(compute_tsh_pop_template.substitute(use_double=use_double),
                     cache_dir=cache_dir)
    compute_tsh_pop = \
                    compute_tsh_pop_mod.get_function('compute_tsh')

    compute_G_pop_mod = \
        SourceModule(compute_G_pop_template.substitute(use_double=use_double),
                     options=['-I', install_headers])
    compute_G_ideal_pop = \
                        compute_G_pop_mod.get_function('compute_G_ideal')
    compute_G_leaky_pop = \
                        compute_G_pop_mod.get_function('compute_G_leaky')
                            
    compute_u_pop_mod = \
        SourceModule(compute_u_pop_template.substitute(use_double=use_double),
                     options=['-I', install_headers])
    compute_u_pop = \
                  compute_u_pop_mod.get_function('compute_u')

    # Total number of interspike intervals per neuron less 1 for each
    # spike train with more than 1 interspike interval:
    Nq = int(np.sum(ns)-np.sum(ns>1))

    # Set up GPUArrays for intermediary data:
    ts_gpu = gpuarray.zeros_like(s_gpu)    
    tsh_gpu = gpuarray.zeros_like(s_gpu)    

    # Note that these arrays are complex to enable use of CUBLAS
    # matrix multiplication functions:
    q_gpu = gpuarray.empty((Nq, 1), complex_type)
    G_gpu = gpuarray.empty((Nq, Nq), complex_type)

    # Get required block/grid sizes:
    block_dim_ts, grid_dim_ts = \
                  cumisc.select_block_grid_sizes(dev, N,
                                                 max_threads_per_block)
    block_dim_q, grid_dim_q = \
                 cumisc.select_block_grid_sizes(dev, q_gpu.shape,
                                                max_threads_per_block)
    block_dim_G, grid_dim_G = \
                 cumisc.select_block_grid_sizes(dev, G_gpu.shape,
                                                max_threads_per_block)
    
    # Launch kernels:
    compute_ts_pop(s_gpu, ns_gpu, ts_gpu, 
                   np.uint32(s_gpu.shape[1]), np.uint32(N),
                   block=block_dim_ts, grid=grid_dim_ts)
    compute_tsh_pop(ts_gpu, ns_gpu, tsh_gpu, 
                    np.uint32(s_gpu.shape[1]), np.uint32(N),
                    block=block_dim_q, grid=grid_dim_q)
    if np.all(np.isinf(R_gpu.get())):
        compute_q_ideal_pop(s_gpu, q_gpu, b_gpu, d_gpu, C_gpu,
                            idx_to_ni_gpu, idx_to_k_gpu,
                            np.uint32(s_gpu.shape[1]),
                            np.uint32(Nq),
                            block=block_dim_q, grid=grid_dim_q)
        compute_G_ideal_pop(ts_gpu, tsh_gpu, G_gpu, float_type(bw),
                            idx_to_ni_gpu, idx_to_k_gpu,
                            np.uint32(Nq),
                            np.uint32(s_gpu.shape[1]),
                            np.uint32(G_gpu.size),
                            block=block_dim_G, grid=grid_dim_G)
    else:
        compute_q_leaky_pop(s_gpu, q_gpu, b_gpu, d_gpu, R_gpu, C_gpu, 
                            idx_to_ni_gpu, idx_to_k_gpu,
                            np.uint32(s_gpu.shape[1]),
                            np.uint32(Nq),
                            block=block_dim_q, grid=grid_dim_q)
        compute_G_leaky_pop(ts_gpu, tsh_gpu, G_gpu, float_type(bw),
                            R_gpu, C_gpu,
                            idx_to_ni_gpu, idx_to_k_gpu,
                            np.uint32(Nq),
                            np.uint32(s_gpu.shape[1]),
                            np.uint32(G_gpu.size),
                            block=block_dim_G, grid=grid_dim_G)

    # Free unneeded variables:
    del ts_gpu, idx_to_k_gpu

    # Compute the reconstruction coefficients:
    c_gpu = culinalg.dot(culinalg.pinv(G_gpu, __pinv_rcond__), q_gpu)

    # Free G, G_inv, and q:
    del G_gpu, q_gpu

    # Allocate arrays needed for reconstruction:
    Nt = int(np.ceil(dur/dt))

    u_rec_gpu = gpuarray.to_gpu(np.zeros(Nt, complex_type))
    ### Replace the above with the following line when the bug in
    # gpuarray.zeros is fixed:
    #u_rec_gpu = gpuarray.zeros(Nt, complex_type)

    # Get required block/grid sizes for constructing u:
    block_dim_t, grid_dim_t = \
                 cumisc.select_block_grid_sizes(dev, Nt, max_threads_per_block)
                                                
    # Reconstruct signal:
    compute_u_pop(u_rec_gpu, c_gpu, tsh_gpu, ns_gpu,
                  float_type(bw), float_type(dt),
                  np.uint32(s_gpu.shape[1]),
                  np.uint32(N),                  
                  np.uint32(Nt), 
                  block=block_dim_t, grid=grid_dim_t)
    u_rec = u_rec_gpu.get()
    
    return np.real(u_rec)
