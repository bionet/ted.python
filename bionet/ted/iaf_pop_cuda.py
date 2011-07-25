#!/usr/bin/env python

from string import Template
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc

from iaf_cuda import iaf_encode_pop

# Get installation location of C headers:
from scikits.cuda import install_headers

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

        // Find the neurons corresponding the current element:
        unsigned int l = idx_to_ni[row];
        unsigned int m = idx_to_ni[col];

        // Find the interspike midpoint entries:
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

        // Find the neurons corresponding the current element:
        unsigned int l = idx_to_ni[row];
        unsigned int m = idx_to_ni[col];

        // Find the interspike midpoint entries:
        unsigned int n = idx_to_k[row];
        unsigned int k = idx_to_k[col];

        FLOAT RC = R[l]*C[l];
        if (ts[INDEX(l,n,s_cols)] < tsh[INDEX(m,k,s_cols)] &&
            tsh[INDEX(m,k,s_cols)] < ts[INDEX(l,n+1,s_cols)]) {
            G[idx] = COMPLEX(0,-1.0/4.0)*EXP((tsh[k]-ts[n+1])/RC)*
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
            c_ind += (ns[m]-1);
        }
        u_rec[idx] = u_temp;
    }
}
""")

def iaf_decode_pop(s_gpu, ns_gpu, dur, dt, bw, b_gpu, d_gpu,
                   R_gpu, C_gpu):
    """
    Population IAF time decoding machine.

    Decode a signal encoded with an ensemble of Integrate-and-Fire
    neurons assuming that the encoded signal is representable in terms
    of sinc kernels.

    Parameters
    ----------

    
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
                            idx_to_ni_gpu, idx_to_k_gpu,
                            np.uint32(Nq),
                            np.uint32(s_gpu.shape[1]),
                            np.uint32(G_gpu.size),
                            block=block_dim_G, grid=grid_dim_G)

    from ipdb import set_trace; set_trace()    
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
    
    
