#!/usr/bin/env python

"""
Time decoding algorithms that make use of the integrate-and-fire
neuron model and the trigonometric polynomial approximation.

- iaf_decode            - IAF time decoding machine.
- iaf_decode_pop        - MISO IAF time decoding machine.
"""

# Copyright (c) 2009-2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

__all__ = ['iaf_decode', 'iaf_decode_pop']

import numpy as np

# Pseudoinverse singular value cutoff:
__pinv_rcond__ = 1e-8

def iaf_decode(s, dur, dt, bw, b, d, R=np.inf, C=1.0, M=5, smoothing=0.0):
    """
    IAF time decoding machine using trigonometric polynomials.

    Decode a finite length signal encoded with an Integrate-and-Fire
    neuron assuming that the encoded signal is representable in terms
    of trigonometric polynomials.

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

    T = 2*np.pi*M/bw
    if T < dur:
        raise ValueError('2*pi*M/bw must exceed the signal length')

    bwM = bw/M
    em = lambda m, t: np.exp(1j*m*bwM*t)

    RC = R*C
    ts = np.cumsum(s)
    F = np.empty((N-1, 2*M+1), complex)
    if np.isinf(R):
        for k in xrange(N-1):
            for m in xrange(-M, M+1):
                if m == 0:
                    F[k, m+M] = s[k+1]
                else:
                    F[k, m+M] = np.conj((em(-m, ts[k+1])-em(-m, ts[k]))/(-1j*m*bwM))
        q = C*d-b*s[1:]
    else:
        for k in xrange(N-1):
            for m in xrange(-M, M+1):
                yk = RC*(1-np.exp(-s[k+1]/RC))
                F[k, m+M] = np.conj((RC*em(-m, ts[k+1])+(yk-RC)*em(-m, ts[k]))/(1-1j*m*bwM*RC))
        q = C*(d+b*R*(np.exp(-s[1:]/RC)-1))

    FH = F.conj().T
    c = np.dot(np.dot(np.linalg.pinv(np.dot(FH,
                                            F)+(N-1)*smoothing*np.eye(2*M+1),
                                     __pinv_rcond__), FH), q)
    t = np.arange(0, dur, dt)
    u_rec = np.zeros(len(t), complex)
    for m in xrange(-M, M+1):
        u_rec += c[m+M]*em(m, t)

    return np.real(u_rec)

def iaf_decode_pop(s_list, dur, dt, bw, b_list, d_list, R_list,
                   C_list, M=5, smoothing=0.0):
    """
    Multi-input single-output IAF time decoding machine.

    Decode a signal encoded with an ensemble of Integrate-and-Fire
    neurons assuming that the encoded signal is representable in terms
    of trigonometric polynomials.

    Parameters
    ----------
    s_list : list of ndarrays of floats
        Signal encoded by an ensemble of encoders. The values represent the
        time between spikes (in s). The number of arrays in the list
        corresponds to the number of encoders in the ensemble.
    dur : float
        Duration of signal (in s).
    dt : float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw : float
        Signal bandwidth (in rad/s).
    b_list : list of floats
        List of encoder biases.
    d_list : list of floats
        List of encoder thresholds.
    R_list : list of floats
        List of encoder neuron resistances.
    C_list : list of floats.
        List of encoder neuron capacitances.
    M : int
        2*M+1 coefficients are used for reconstructing the signal.
    smoothing : float
        Smoothing parameter.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.

    Notes
    -----
    The number of spikes contributed by each neuron may differ from the
    number contributed by other neurons.
    """

    # Number of neurons:
    N = len(s_list)
    if not N:
        raise ValueError('no spike data given')

    T = 2*np.pi*M/bw
    if T < dur:
        raise ValueError('2*pi*M/bw must exceed the signal length')

    bwM = bw/M
    em = lambda m, t: np.exp(1j*m*bwM*t)

    # Number of interspike intervals per neuron:
    ns = np.array(map(len, s_list))

    # Compute the spike times:
    ts_list = map(np.cumsum, s_list)

    # Indices for accessing subblocks of the reconstruction matrix:
    Fi = np.cumsum(np.hstack([0, ns-1]))

    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    Nq = np.sum(ns)-np.sum(ns>1)
    F = np.empty((Nq, 2*M+1), complex)
    q = np.empty((Nq, 1), np.float)
    if all(np.isinf(R_list)):
        for i in xrange(N):
            ts = ts_list[i]
            F_temp = np.empty((ns[i]-1, 2*M+1), complex)
            q_temp = np.empty((ns[i], 1), np.float)
            for k in xrange(ns[i]-1):
                for m in xrange(-M, M+1):
                    if m == 0:
                        F_temp[k, m+M] = s_list[i][k+1]
                    else:
                        F_temp[k, m+M] = (em(m, ts[k+1])- \
                                          em(m, ts[k]))/(1j*m*bwM)

            F[Fi[i]:Fi[i+1], :] = F_temp
            q[Fi[i]:Fi[i+1], 0] = \
                C_list[i]*d_list[i]-b_list[i]*s_list[i][1:]
    else:
        for i in xrange(N):
            ts = ts_list[i]
            F_temp = np.empty((ns[i]-1, 2*M+1), complex)
            q_temp = np.empty((ns[i], 1), np.float)
            RC = R_list[i]*C_list[i]
            for k in xrange(ns[i]-1):
                for m in xrange(-M, M+1):
                    if m == 0:
                        F_temp[k, m+M] = (np.exp(ts[k+1]/RC)-np.exp(ts[k]/RC))* \
                                         np.exp(-ts[k+1]/RC)*RC
                    else:
                        x = 1j*m*bwM+1/RC
                        F_temp[k, m+M] = (np.exp(ts[k+1]*x)-np.exp(ts[k]*x))* \
                                         np.exp(-ts[k+1]/RC)/x

            F[Fi[i]:Fi[i+1], :] = F_temp
            q[Fi[i]:Fi[i+1], 0] = \
                C_list[i]*d_list[i]-b_list[i]*RC*(1-np.exp(-s_list[i][1:]/RC))

    FH = F.conj().T
    c = np.dot(np.dot(np.linalg.pinv(np.dot(FH, F)+(N-1)*smoothing*np.eye(2*M+1), __pinv_rcond__), FH), q)

    t = np.arange(0, dur, dt)
    u_rec = np.zeros(len(t), complex)
    for m in xrange(-M, M+1):
        u_rec += c[m+M]*em(m, t)

    return np.real(u_rec)
