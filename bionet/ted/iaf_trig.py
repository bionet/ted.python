#!/usr/bin/env python

"""
Time decoding algorithms that use the trigonometric polynomial
approximation.
"""

from numpy import arange, conj, cumsum, dot, empty, exp, eye, inf, \
     isinf, pi, real, zeros     
from numpy.linalg import pinv

def iaf_decode_trig(s, dur, dt, bw, b, d, R=inf, C=1.0, M=5, smoothing=0.0):
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

    T = 2*pi*M/bw
    if T < dur:
        raise ValueError('2*pi*M/bw must exceed the signal length')

    bwM = bw/M
    em = lambda m, t: exp(1j*m*bwM*t)

    RC = R*C
    ts = cumsum(s)
    G = empty((N-1, 2*M+1), complex)
    if isinf(R):        
        for k in xrange(N-1):
            for m in xrange(-M, M+1):
                if m == 0:
                    G[k, m+M] = s[k+1]
                else:
                    G[k, m+M] = conj((em(-m, ts[k+1])-em(-m, ts[k]))/(-1j*m*bwM)) 
        q = C*d-b*s[1:]        
    else:
        for k in xrange(N-1):
            for m in xrange(-M, M+1):
                yk = RC*(1-exp(-s[k+1]/RC))
                G[k, m+M] = conj((RC*em(-m, ts[k+1])+(yk-RC)*em(-m, ts[k]))/(1-1j*m*bwM*RC))
        q = C*(d+b*R*(exp(-s[1:]/RC)-1))

    GH = G.conj().T
    c = dot(dot(pinv(dot(GH, G)+(N-1)*smoothing*eye(2*M+1)), GH), q)
    t = arange(0, dur, dt)
    u_rec = zeros(len(t), complex)
    for m in xrange(-M, M+1):
        u_rec += c[m+M]*em(m, t)

    return real(u_rec)
