#!/usr/bin/env python

"""
Time decoding algorithms that use the trigonometric polynomial
approximation.

- iaf_decode            - IAF time decoding machine.
- iaf_decode_pop        - MISO IAF time decoding machine.

These functions make use of CUDA.

"""

from string import Template
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import scikits.cuda.misc as cumisc
import scikits.cuda.linalg as culinalg

from iaf_cuda import iaf_encode, iaf_encode_pop

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
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
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
    unsigned int idx = blockIdx.y*blockDim.x*gridDim.x+
                       blockIdx.x*blockDim.x+threadIdx.x;
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

def iaf_decode(s, dur, dt, bw, b, d, R=np.inf, C=1.0, M=5, smoothing=0.0):
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
        __pinv_rcond__ = 1e-4
    elif float_type == np.float64:
        use_double = 1
        complex_type = np.complex128
        __pinv_rcond__ = 1e-8
    else:
        raise ValueError('unsupported data type')
        
    T = 2*np.pi*M/bw
    if T < dur:
        raise ValueError('2*pi*M/bw must exceed the signal length')

    dev = cumisc.get_current_device()
    
    # Prepare kernels:
    cache_dir = None
    compute_q_mod = \
                  SourceModule(compute_q_template.substitute(use_double=use_double),
                               cache_dir=cache_dir)
    compute_q_ideal = compute_q_mod.get_function('compute_q_ideal')
    compute_q_leaky = compute_q_mod.get_function('compute_q_leaky')

    compute_F_mod = \
                  SourceModule(compute_F_template.substitute(use_double=use_double),
                               cache_dir=cache_dir)
    compute_F_ideal = compute_F_mod.get_function('compute_F_ideal')
    compute_F_leaky = compute_F_mod.get_function('compute_F_leaky')

    compute_u_mod = \
                  SourceModule(compute_u_template.substitute(use_double=use_double),
                               cache_dir=cache_dir)
    compute_u = compute_u_mod.get_function('compute_u')

    # Load data into GPU memory:
    s_gpu = gpuarray.to_gpu(s)

    # XXX: Eventually replace this with a PyCUDA equivalent
    ts = np.cumsum(s)
    ts_gpu = gpuarray.to_gpu(ts)

    # Set up GPUArrays for intermediary data. Note that all of the
    # arrays are complex to facilitate use of CUBLAS matrix
    # multiplication functions:
    q_gpu = gpuarray.empty((N-1, 1), complex_type)
    F_gpu = gpuarray.empty((N-1, 2*M+1), complex_type)

    # Get required block/grid sizes; use a smaller block size than the
    # maximum to prevent the kernels from using too many registers:
    max_threads_per_block = 256
    block_dim_s, grid_dim_s = cumisc.select_block_grid_sizes(dev,
                                                             q_gpu.shape,
                                                             max_threads_per_block)
    block_dim_F, grid_dim_F = cumisc.select_block_grid_sizes(dev,
                                                             F_gpu.shape,
                                                             max_threads_per_block)
    if np.isinf(R):
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

    # Compute the product of F^H and q first so that q
    # can be dropped from memory:
    FHq_gpu = culinalg.dot(F_gpu, q_gpu, 'c')
    del q_gpu
    
    if smoothing == 0:
        c_gpu = culinalg.dot(culinalg.pinv(culinalg.dot(F_gpu, F_gpu, 'c'),
                                           __pinv_rcond__),
                             FHq_gpu)
    else:
        c_gpu = culinalg.dot(culinalg.pinv(culinalg.dot(F_gpu, F_gpu, 'c')+
                                           (N-1)*smoothing*culinalg.eye(2*M+1,
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

compute_ts_pop_template = Template("""
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

compute_F_pop_template = Template("""
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

compute_q_pop_template = Template("""
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

compute_u_pop_template = Template("""
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
        __pinv_rcond__ = 1e-4
    elif float_type == np.float64:
        use_double = 1
        complex_type = np.complex128
        __pinv_rcond__ = 1e-8
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
    compute_ts_pop_mod = SourceModule(compute_ts_pop_template.substitute(use_double=use_double),
                                  cache_dir=cache_dir)
    compute_ts_pop = compute_ts_pop_mod.get_function('compute_ts')
    
    compute_q_pop_mod = \
                      SourceModule(compute_q_pop_template.substitute(use_double=use_double),
                                   cache_dir=cache_dir)
    compute_q_pop_ideal = compute_q_pop_mod.get_function('compute_q_ideal')
    compute_q_pop_leaky = compute_q_pop_mod.get_function('compute_q_leaky')

    compute_F_pop_mod = \
                  SourceModule(compute_F_pop_template.substitute(use_double=use_double),
                               cache_dir=cache_dir,
                               options=['-I', install_headers])
    compute_F_pop_ideal = compute_F_pop_mod.get_function('compute_F_ideal')
    compute_F_pop_leaky = compute_F_pop_mod.get_function('compute_F_leaky')

    compute_u_pop_mod = \
                      SourceModule(compute_u_pop_template.substitute(use_double=use_double),
                                   cache_dir=cache_dir,
                                   options=['-I', install_headers])
    compute_u_pop = compute_u_pop_mod.get_function('compute_u')
    
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
    compute_ts_pop(s_gpu, ns_gpu, ts_gpu, np.uint32(s_gpu.shape[1]),
                   np.uint32(N),
                   block=block_dim_ts, grid=grid_dim_ts)
    if np.all(np.isinf(R_gpu.get())):
        compute_q_pop_ideal(s_gpu, q_gpu,
                            b_gpu, d_gpu, C_gpu,
                            idx_to_ni_gpu, idx_to_k_gpu,
                            np.uint32(s_gpu.shape[1]),
                            np.uint32(Nq),
                            block=block_dim_q, grid=grid_dim_q)
        compute_F_pop_ideal(s_gpu, ts_gpu, F_gpu,
                            float_type(bw),
                            idx_to_ni_gpu, idx_to_k_gpu,
                            np.int32(M), np.uint32(s_gpu.shape[1]),
                            np.uint32(F_gpu.size),
                            block=block_dim_F, grid=grid_dim_F)
    else:
        compute_q_pop_leaky(s_gpu, q_gpu,
                            b_gpu, d_gpu,
                            R_gpu, C_gpu,
                            idx_to_ni_gpu, idx_to_k_gpu,
                            np.uint32(s_gpu.shape[1]),
                            np.uint32(Nq),
                            block=block_dim_q, grid=grid_dim_q)
        compute_F_pop_leaky(s_gpu, ts_gpu, F_gpu,
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
    compute_u_pop(u_rec_gpu, c_gpu, float_type(bw),
                  float_type(dt),
                  np.int32(M),
                  np.uint32(Nt),
                  block=block_dim_t, grid=grid_dim_t)

    return np.real(u_rec_gpu.get())
