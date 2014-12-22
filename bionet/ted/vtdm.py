#!/usr/bin/env python

"""
Block-based time decoding algorithm used by real-time time decoding algorithm.
"""

# Copyright (c) 2009-2014, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

__all__ = ['asdm_decode_vander', 'asdm_decode_vander_ins',
           'iaf_decode_vander']

import numpy as np

import bionet.utils.numpy_extras as ne
import bionet.ted.bpa as bpa

def asdm_decode_vander(s, dur, dt, bw, b, d, k, sgn=-1):
    """
    Asynchronous Sigma-Delta Modulator time decoding machine that uses
    BPA.

    Decode a finite length signal encoded with an Asynchronous
    Sigma-Delta Modulator by efficiently solving a Vandermonde system
    using the Bjork-Pereyra Algorithm.

    Parameters
    ----------
    s: array_like of floats
        Encoded signal. The values represent the time between spikes (in s).
    dur: float
        Duration of signal (in s).
    dt: float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw: float
        Signal bandwidth (in rad/s).
    b: float
        Encoder bias.
    d: float
        Encoder threshold.
    k: float
        Encoder integration constant.
    sgn: {-1, 1}
        Sign of first spike.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.
    """

    # Since the compensation principle uses the differences between
    # spikes, the last spike must effectively be dropped:
    ns = len(s)-1
    n = ns-1               # corresponds to N in Prof. Lazar's paper

    # Cast s to an ndarray to permit ndarray operations:
    s = np.asarray(s)

    # Compute the spike times:
    ts = np.cumsum(s)

    # Create the vectors and matricies needed to obtain the
    # reconstruction coefficients:
    z = np.exp(1j*2*bw*ts[:-1]/n)

    V = np.fliplr(np.vander(z))  # pecularity of numpy's vander() function
    P = np.triu(np.ones((ns, ns), np.float))
    D = np.diag(np.exp(1j*bw*ts[:-1]))

    # Compute the quanta:
    if sgn == -1:
        q = np.asarray([(-1)**i for i in xrange(0, ns)])*(2*k*d-b*s[1:])
    else:
        q = np.asarray([(-1)**i for i in xrange(1, ns+1)])*(2*k*d-b*s[1:])

    # Obtain the reconstruction coefficients by solving the
    # Vandermonde system using BPA:
    d = bpa.bpa(V, ne.mdot(D, P, q[:, np.newaxis]))

    # Reconstruct the signal:
    t = np.arange(0, dur, dt)
    u_rec = np.zeros(len(t), np.complex)
    for i in xrange(ns):
        c = 1j*(bw-i*2*bw/n)
        u_rec += c*d[i]*np.exp(-c*t)

    return np.real(u_rec)

def asdm_decode_vander_ins(s, dur, dt, bw, b, sgn=-1):
    """
    Threshold-insensitive ASDM time decoding machine that uses BPA.

    Decode a finite length signal encoded with an Asynchronous
    Sigma-Delta Modulator by efficiently solving a
    threshold-insensitive Vandermonde system using the Bjork-Pereyra
    Algorithm.

    Parameters
    ----------
    s: array_like of floats
        Encoded signal. The values represent the time between spikes (in s).
    dur: float
        Duration of signal (in s).
    dt: float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw: float
        Signal bandwidth (in rad/s).
    b: float
        Encoder bias.
    sgn: {-1, 1}
        Sign of first spike.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.
    """

    # Since the compensation principle uses the differences between
    # spikes, the last spike in s must effectively be dropped:
    ns = len(s)-1
    n = ns-1               # corresponds to N in Prof. Lazar's paper

    # Cast s to an ndarray to permit ndarray operations:
    s = np.asarray(s)

    # Compute the spike times:
    ts = np.cumsum(s)

    # Create the vectors and matricies needed to obtain the
    # reconstruction coefficients:
    z = np.exp(1j*2*bw*ts[:-1]/n)
    V = np.fliplr(np.vander(z))  # pecularity of numpy's vander() function
    D = np.diag(np.exp(1j*bw*ts[:-1]))
    P = np.triu(np.ones((ns, ns), np.float))

    a = np.zeros(ns, np.float)
    a[::-2] = 1.0
    a = a[:, np.newaxis]      # column vector

    bh = np.zeros(ns, np.float)
    bh[-1] = 1.0
    bh = bh[np.newaxis]       # row vector

    ex = np.ones(ns, np.float)
    if sgn == -1:
        ex[0::2] = -1.0
    else:
        ex[1::2] = -1.0
    r = (ex*s[1:])[:, np.newaxis]

    # Solve the Vandermonde systems using BPA:
    ## Observation: constructing P-dot(a,bh) directly without
    ## creating P, a, and bh separately does not speed this up
    x = bpa.bpa(V, ne.mdot(D, P-np.dot(a, bh), r))
    y = bpa.bpa(V, np.dot(D, a))

    # Compute the coefficients:
    d = b*(x-ne.mdot(y, np.conj(y.T), x)/np.dot(np.conj(y.T), y))

    # Reconstruct the signal:
    t = np.arange(0, dur, dt)
    u_rec = np.zeros(len(t), np.complex)
    for i in xrange(ns):
        c = 1j*(bw-i*2*bw/n)
        u_rec += c*d[i]*np.exp(-c*t)

    return np.real(u_rec)

def iaf_decode_vander(s, dur, dt, bw, b, d, R, C):
    """
    IAF time decoding machine that uses BPA.

    Decode a finite length signal encoded with an Integrate-and-Fire
    neuron by efficiently solving a Vandermonde system using the
    Bjork-Pereyra Algorithm.

    Parameters
    ----------
    s: array_like of floats
        Encoded signal. The values represent the time between spikes (in s).
    dur: float
        Duration of signal (in s).
    dt: float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw: float
        Signal bandwidth (in rad/s).
    b: float
        Encoder bias.
    d: float
        Encoder threshold.
    R: float
        Neuron resistance.
    C: float
        Neuron capacitance.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.
    """

    # Since the compensation principle uses the differences between
    # spikes, the last spike must effectively be dropped:
    ns = len(s)-1
    n = ns-1               # corresponds to N in Prof. Lazar's paper

    # Cast s to an ndarray to permit ndarray operations:
    s = np.asarray(s)

    # Compute the spike times:
    ts = np.cumsum(s)

    # Create the vectors and matricies needed to obtain the
    # reconstruction coefficients:
    z = np.exp(1j*2*bw*ts[:-1]/n)

    V = np.fliplr(np.vander(z))  # pecularity of numpy's vander() function
    P = np.triu(np.ones((ns, ns), np.float))
    D = np.diag(np.exp(1j*bw*ts[:-1]))

    # Compute the quanta:
    if np.isinf(R):
        q = np.asarray(C*d-b*s[1:])
    else:
        q = np.asarray(C*(d+b*R*(np.exp(-s[1:]/(R*C))-1)))

    # Obtain the reconstruction coefficients by solving the
    # Vandermonde system using BPA:
    d = bpa.bpa(V, ne.mdot(D, P, q[:, np.newaxis]))

    # Reconstruct the signal:
    t = np.arange(0, dur, dt)
    u_rec = np.zeros(len(t), np.complex)
    for i in xrange(ns):
        c = 1j*(bw-i*2*bw/n)
        u_rec += c*d[i]*np.exp(-c*t)

    return np.real(u_rec)
