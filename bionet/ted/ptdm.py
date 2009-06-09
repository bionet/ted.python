#!/usr/bin/env python

"""
Decode a signal encoded by several time encoding machines.
"""

from numpy import array, cumsum, empty, zeros, float, sum, dot, pi, sinc, \
     linspace, hstack
from numpy.linalg import pinv
from scipy.special import sici

def pop_decode(s_list, dur, dt, bw, b_list, d_list, k_list):
    """Decode a finite length signal encoded by an ensemble of asynchronous
    sigma-delta modulators. 

    Parameters
    ----------
    s_list: list of numpy arrays of floats
        Signal encoded by an ensemble of encoders. The values represent the
        time between spikes (in s). The number of arrays in the list
        corresponds to the number of encoders in the ensemble.
    dur: float
        Duration of signal (in s).
    dt: float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw: float
        Signal bandwidth (in rad/s).
    b_list: list of floats
        List of encoder biases.
    d_list: list of floats
        List of encoder thresholds.
    k_list: list of floats
        List of encoder integration constants.            
    
    Notes
    -----
    The number of spikes contributed by each neuron may differ from the
    number contributed by other neurons.
    """

    M = len(s_list)
    if not M:
        raise ValueError('no spike data given')

    bwpi = bw/pi
    
    # Compute the midpoints between spikes:
    ts_list = map(cumsum,s_list)
    tsh_list = map(lambda ts:(ts[0:-1]+ts[1:])/2,ts_list)

    # Compute number of spikes in each spike list:
    Ns_list = map(len,ts_list)
    Nsh_list = map(len,tsh_list)

    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    Nsh_sum = sum(Nsh_list)
    G = empty((Nsh_sum,Nsh_sum),float)
    q = empty((Nsh_sum,1),float)
    for l in xrange(M):
        for m in xrange(M):
            G_block = empty((Nsh_list[l],Nsh_list[m]),float)

            # Compute the values for all of the sincs so that they
            # do not need to each be recomputed when determining
            # the integrals between spike times:
            for k in xrange(Nsh_list[m]):
                temp = sici(bw*(ts_list[l]-tsh_list[m][k]))[0]/pi
                for n in xrange(Nsh_list[l]):
                    G_block[n,k] = temp[n+1]-temp[n]

            G[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]),
              sum(Nsh_list[:m]):sum(Nsh_list[:m+1])] = G_block

        # Compute the quanta:
        q[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]),0] = \
                       array([(-1)**i for i in xrange(1,Nsh_list[l]+1)])* \
                       (2*k_list[l]*d_list[l]-b_list[l]*s_list[l][1:])

    # Compute the reconstruction coefficients:
    c = dot(pinv(G),q)

    # Reconstruct the signal using the coefficients:
    Nt = int(dur/dt)
    t = linspace(0,dur,Nt)
    u_rec = zeros(Nt,float)
    for m in xrange(M):
        for k in xrange(Nsh_list[m]):
            u_rec += sinc(bwpi*(t-tsh_list[m][k]))*bwpi*c[sum(Nsh_list[:m])+k,0]
    return u_rec

def pop_decode_ins(s_list, dur, dt, bw, b_list):
    """Decode a finite length signal encoded by an ensemble of asynchronous
    sigma-delta modulators using a threshold-insensitive recovery algorithm.

    Parameters
    ----------
    s_list: list of numpy arrays of floats
        Signal encoded by an ensemble of encoders. The values represent the
        time between spikes (in s). The number of arrays in the list
        corresponds to the number of encoders in the ensemble.
    dur: float
        Duration of signal (in s).
    dt: float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw: float
        Signal bandwidth (in rad/s).
    b_list: list of floats
        List of encoder biases.

    Notes
    -----
    The number of spikes contributed by each neuron may differ from the
    number contributed by other neurons.
    """

    M = len(s_list)
    if not M:
        raise ValueError('no spike data given')

    bwpi = bw/pi
    
    # Compute the midpoints between spikes:
    ts_list = map(cumsum,s_list)
    tsh_list = map(lambda ts:(ts[0:-1]+ts[1:])/2,ts_list)

    # Compute number of spikes in each spike list:
    Ns_list = map(len,ts_list)
    Nsh_list = map(lambda x: len(x)-1,tsh_list)
    
    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    Nsh_sum = sum(Nsh_list)
    G = empty((Nsh_sum,Nsh_sum),float)
    Bq = empty((Nsh_sum,1),float)
    for l in xrange(M):
        for m in xrange(M):
            G_block = empty((Nsh_list[l],Nsh_list[m]),float)

            # Compute the values for all of the sincs so that they
            # do not need to each be recomputed when determining
            # the integrals between spike times:
            for k in xrange(Nsh_list[m]):
                temp = sici(bw*(ts_list[l]-tsh_list[m][k]))[0]/pi
                for n in xrange(Nsh_list[l]):
                    G_block[n,k] = temp[n+2]-temp[n]

            G[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]),
              sum(Nsh_list[:m]):sum(Nsh_list[:m+1])] = G_block

        # Compute the quanta:
        Bq[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]),0] = \
                       array([(-1)**i for i in xrange(1,Nsh_list[l]+1)])* \
                       b_list[l]*(s_list[l][2:]-s_list[l][1:-1])

    # Compute the reconstruction coefficients:
    c = dot(pinv(G),Bq)

    # Reconstruct the signal using the coefficients:
    Nt = int(dur/dt)
    t = linspace(0,dur,Nt)
    u_rec = zeros(Nt,float)
    for m in xrange(M):
        for k in xrange(Nsh_list[m]):
            u_rec += sinc(bwpi*(t-tsh_list[m][k]))*bwpi*c[sum(Nsh_list[:m])+k,0]
    return u_rec
