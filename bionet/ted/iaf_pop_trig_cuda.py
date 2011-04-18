#!/usr/bin/env python

"""
Population time decoding algorithms that use the trigonometric polynomial
approximation.

These functions make use of CUDA.

"""

from string import Template
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import scikits.cuda.misc as cumisc
import scikits.cuda.linalg as culinalg
from scikits.cuda import install_headers

from prodtrans import prodtrans

# Pseudoinverse singular value cutoff:
__pinv_rcond__ = 1e-8

# Kernel template for performing ideal/leaky time encoding a
# 1D signal using N encoders:
iaf_encode_pop_template = Template("""
#if ${use_double}
#define FLOAT double
#define EXP(x) exp(x)
#else
#define FLOAT float
#define EXP(x) expf(x)
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
                y_curr = y_curr*EXP(-dt/RC_curr)+
                         R_curr*(1-EXP(-dt/RC_curr))*(b_curr+u_curr);                        
            
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

    Encode a finite length signals with a population of Integrate-and-Fire
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
                                # options=['--ptxas-options=-v'],
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

compute_ts_template = Template("""
#if ${use_double}
#define FLOAT double
#else
#define FLOAT float
#endif

// Macro for accessing multidimensional arrays with cols columns by
// linear index:
#define INDEX(row,col,cols) row*cols+col

// Assumes that ts is initially filled with zeros
// s: interspike intervals; shape(s) == (N, s_cols)
// ns: lengths of spike trains
// ts: computed spike times
// s_cols: number of columns in s
// N: number of rows in s
__global__ void compute_ts(FLOAT *s, unsigned int *ns, FLOAT *ts,
                           unsigned int s_cols,
                           unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    if (idx < N) {
        FLOAT temp = 0;
        unsigned int j;
        for (unsigned int i = 0; i < ns[idx]; i++) {
            j = INDEX(idx,i,s_cols);
            temp += s[j];
            ts[j] = temp;
        }
    }
}                      
""")

compute_F_template = Template("""
#include <pycuda/pycuda-complex.hpp>
#include "cuConstants.h"

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#define SQRT(x) sqrt(x)
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#define SQRT(x) sqrtf(x)
#endif

// Macro for accessing multidimensional arrays with cols columns by
// linear index:
#define INDEX(row,col,cols) row*cols+col

#define EM(m,t,bw,M) exp(COMPLEX(0, m*bw*t/M))

// s: interspike intervals
// ts: spike times
// F: computed reconstruction matrix
// bw: bandwidth (rad/s)
// R: resistances
// C: capacitances
// idx_to_ni: map from linear index to neuron number
// idx_to_k: map from linear index to spike time index
// M: trigonometric polynomial order
// s_cols: number of columns in s
// N: Nq*(2*M+1), where Nq is the length of q
__global__ void compute_F_ideal(FLOAT *s, FLOAT *ts, COMPLEX *F, FLOAT bw, 
                                unsigned int *idx_to_ni,
                                unsigned int *idx_to_k,
                                int M,
                                unsigned int s_cols,
                                unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int row = idx/(2*M+1);
    int m = (idx%(2*M+1))-M;
    
    if (idx < N) {
        unsigned int ni = idx_to_ni[row];
        unsigned int k = idx_to_k[row];

        FLOAT ts_curr = ts[INDEX(ni,k,s_cols)];
        FLOAT ts_next = ts[INDEX(ni,k+1,s_cols)];

        if (m == 0) {
            F[idx] = COMPLEX(s[INDEX(ni,k+1,s_cols)]);
        } else {
            F[idx] = (EM(m,ts_next,bw,M)-EM(m,ts_curr,bw,M))/COMPLEX(0,m*bw/M);
        }
    }
}                          

__global__ void compute_F_leaky(FLOAT *s, FLOAT *ts, COMPLEX *F, FLOAT bw,
                                FLOAT *R, FLOAT *C, 
                                unsigned int *idx_to_ni,
                                unsigned int *idx_to_k,
                                int M,
                                unsigned int s_cols,
                                unsigned int N) {
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int row = idx/(2*M+1);
    int m = (idx%(2*M+1))-M;
    
    if (idx < N) {
        unsigned int ni = idx_to_ni[row];
        unsigned int k = idx_to_k[row];

        FLOAT ts_curr = ts[INDEX(ni,k,s_cols)];
        FLOAT ts_next = ts[INDEX(ni,k+1,s_cols)];
        FLOAT RC = R[ni]*C[ni];
        
        if (m == 0) {
            F[idx] = COMPLEX((exp(ts_next/RC)-exp(ts_curr/RC))*
                     exp(-ts_next/RC)*RC);
        } else {
            COMPLEX x = COMPLEX(1/RC,m*bw/M);
            F[idx] = (exp(ts_next*x)-exp(ts_curr*x))*exp(-ts_next/RC)/x;
        }    
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

// Macro for accessing multidimensional arrays with cols columns by
// linear index:
#define INDEX(row,col,cols) row*cols+col

// s: interspike intervals
// q: computed q array
// b: neuron biases
// d: neuron thresholds
// R: neuron resistances
// C: neuron capacitances
// idx_to_ni: map from linear index to neuron index
// idx_to_k: map from linear index to interspike interval index
// s_cols: number of columns in s
// N: the sum of the number of interspike intervals per neuron less 1
// for each neuron with more than 1 spike time
__global__ void compute_q_ideal(FLOAT *s, COMPLEX *q, FLOAT *b,
                                FLOAT *d, FLOAT *C,
                                unsigned int *idx_to_ni,
                                unsigned int *idx_to_k,
                                unsigned int s_cols,
                                unsigned int N) {                          
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N) {
        unsigned int ni = idx_to_ni[idx];
        unsigned int k = idx_to_k[idx];

        q[idx] = COMPLEX(C[ni]*d[ni]-b[ni]*s[INDEX(ni,k+1,s_cols)]);
    }
}

__global__ void compute_q_leaky(FLOAT *s, COMPLEX *q, FLOAT *b,
                                FLOAT *d, FLOAT *R, FLOAT *C,
                                unsigned int *idx_to_ni,
                                unsigned int *idx_to_k,
                                unsigned int s_cols,
                                unsigned int N) {                          
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;

    if (idx < N) {
        unsigned int ni = idx_to_ni[idx];
        unsigned int k = idx_to_k[idx];
    
        q[idx] = COMPLEX(C[ni]*(d[ni]+
                 b[ni]*R[ni]*(exp(-s[INDEX(ni,k+1,s_cols)]/(R[ni]*C[ni]))-1)));
    }
}
""")

compute_u_template = Template("""
#include <pycuda/pycuda-complex.hpp>
#include "cuConstants.h"

#if ${use_double}
#define FLOAT double
#define COMPLEX pycuda::complex<double>
#define SQRT(x) sqrt(x)
#else
#define FLOAT float
#define COMPLEX pycuda::complex<float>
#define SQRT(x) sqrtf(x)
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
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
                       
    if (idx < Nt) {
        COMPLEX u_temp = COMPLEX(0);        
        for (int m = -M; m <= M+1; m++) {
            u_temp += c[m+M]*EM(m,idx*dt,bw,M);
        }
        u_rec[idx] = u_temp;
    }
}
""")

def iaf_decode_pop(s_gpu, ns_gpu, dur, dt, bw, b_gpu, d_gpu, R_gpu,
                   C_gpu, M=5, smoothing=0.0):
    """
    Population IAF time decoding machine.
    
    Decode a signal encoded with an ensemble of Integrate-and-Fire
    neurons assuming that the encoded signal is representable in terms
    of trigonometric polynomials.

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
    M : int
        2*M+1 coefficients are used for reconstructing the signal.
    smoothing : float
        Smoothing parameter.

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
    elif float_type == np.float64:
        use_double = 1
        complex_type = np.complex128
    else:
        raise ValueError('unsupported data type')

    N = s_gpu.shape[0]
    if not N:
        raise ValueError('no spike data given')
    if (ns_gpu.size != N) or (b_gpu.size != N) or (d_gpu.size != N) or \
       (R_gpu.size != N) or (C_gpu.size != N):
        raise ValueError('parameter arrays must be of same length')
    
    T = 2*np.pi*M/bw
    if T < dur:
        raise ValueError('2*pi*M/bw must exceed the signal length')
                    
    # Map CUDA index to neuron index and interspike interval index:
    ns = ns_gpu.get()
    idx_to_ni, idx_to_k = _compute_idx_map(ns)
    idx_to_ni_gpu = gpuarray.to_gpu(idx_to_ni)
    idx_to_k_gpu = gpuarray.to_gpu(idx_to_k)

    dev = cumisc.get_current_device()

    # Use a smaller block size than the maximum to prevent the kernels
    # from using too many registers:
    max_threads_per_block = 256

    # Prepare kernels:
    cache_dir = None
    compute_ts_mod = SourceModule(compute_ts_template.substitute(use_double=use_double),
                                  cache_dir=cache_dir)
    compute_ts = compute_ts_mod.get_function('compute_ts')
    
    compute_q_mod = \
                  SourceModule(compute_q_template.substitute(use_double=use_double),
                               cache_dir=cache_dir)
    compute_q_ideal = compute_q_mod.get_function('compute_q_ideal')
    compute_q_leaky = compute_q_mod.get_function('compute_q_leaky')

    compute_F_mod = \
                  SourceModule(compute_F_template.substitute(use_double=use_double),
                               cache_dir=cache_dir,
                               options=['-I', install_headers])
    compute_F_ideal = compute_F_mod.get_function('compute_F_ideal')
    compute_F_leaky = compute_F_mod.get_function('compute_F_leaky')

    compute_u_mod = \
                  SourceModule(compute_u_template.substitute(use_double=use_double),
                               cache_dir=cache_dir,
                               options=['-I', install_headers])
    compute_u = compute_u_mod.get_function('compute_u')
    
    # Total number of interspike intervals per neuron less 1 for each
    # spike train with more than
    Nq = int(np.sum(ns)-np.sum(ns>1))
    
    # Set up GPUArrays for intermediary data: 
    ts_gpu = gpuarray.zeros_like(s_gpu)

    # Note that these arrays are complex to enable use of CUBLAS
    # matrix multiplication functions:
    q_gpu = gpuarray.empty((Nq, 1), complex_type)
    F_gpu = gpuarray.empty((Nq, 2*M+1), complex_type) 

    # Get required block/grid sizes:
    block_dim_ts, grid_dim_ts = \
                  cumisc.select_block_grid_sizes(dev, N,
                                                 max_threads_per_block)
    block_dim_q, grid_dim_q = \
                 cumisc.select_block_grid_sizes(dev, q_gpu.shape,
                                                max_threads_per_block)
    block_dim_F, grid_dim_F = \
                 cumisc.select_block_grid_sizes(dev, F_gpu.shape,
                                                max_threads_per_block)

    # Launch kernels:
    compute_ts(s_gpu, ns_gpu, ts_gpu, np.uint32(s_gpu.shape[1]),
               np.uint32(N),
               block=block_dim_ts, grid=grid_dim_ts)
    if np.all(np.isinf(R_gpu.get())):
        compute_q_ideal(s_gpu, q_gpu,
                        b_gpu, d_gpu, C_gpu,
                        idx_to_ni_gpu, idx_to_k_gpu,
                        np.uint32(s_gpu.shape[1]),
                        np.uint32(Nq),
                        block=block_dim_q, grid=grid_dim_q)
        compute_F_ideal(s_gpu, ts_gpu, F_gpu,
                        float_type(bw),
                        idx_to_ni_gpu, idx_to_k_gpu,
                        np.int32(M), np.uint32(s_gpu.shape[1]),
                        np.uint32(F_gpu.size),
                        block=block_dim_F, grid=grid_dim_F)
    else:
        compute_q_leaky(s_gpu, q_gpu,
                        b_gpu, d_gpu,
                        R_gpu, C_gpu,
                        idx_to_ni_gpu, idx_to_k_gpu,
                        np.uint32(s_gpu.shape[1]),
                        np.uint32(Nq),
                        block=block_dim_q, grid=grid_dim_q)
        compute_F_leaky(s_gpu, ts_gpu, F_gpu,
                        float_type(bw), R_gpu, C_gpu,
                        idx_to_ni_gpu, idx_to_k_gpu,
                        np.int32(M), np.uint32(s_gpu.shape[1]),
                        np.uint32(F_gpu.size),
                        block=block_dim_F, grid=grid_dim_F)

    # Free unneeded variables:
    del s_gpu, ts_gpu, idx_to_ni_gpu, idx_to_k_gpu

    # Compute the product of F^H and q first so that both F^H and q
    # can be dropped from memory:
    FH_gpu = culinalg.hermitian(F_gpu)
    FHq_gpu = culinalg.dot(FH_gpu, q_gpu)
    del FH_gpu, q_gpu

    if smoothing == 0:
        c_gpu = culinalg.dot(culinalg.pinv(culinalg.dot(F_gpu, F_gpu, 'c'),
                                           __pinv_rcond__), 
                             FHq_gpu)
    else:
        c_gpu = culinalg.dot(culinalg.pinv(culinalg.dot(F_gpu, F_gpu, 'c')+
                                           np.sum(ns)*smoothing*culinalg.eye(2*M+1,
                                                                        float_type),
                                           __pinv_rcond__),   
                             FHq_gpu)
        
    # Allocate array for reconstructed signal:
    Nt = int(np.ceil(dur/dt))
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

    return np.real(u_rec_gpu.get())
