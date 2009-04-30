#!/usr/bin/env python

"""
Time encoding and decoding algorithms that make use of the
asynchronous sigma-delta modulator.
"""

__all__ = ['asdm_recoverable','asdm_encode','asdm_decode','asdm_decode_ins']
           
from numpy import max, abs, size, zeros, ones, float, pi, array,\
     shape, dot, eye, sinc, hstack, newaxis, cumsum, linspace, empty,\
     diag, diff, eye
from numpy.linalg import inv, pinv
from scipy.signal import resample

# The sici() function is used to obtain the values in the decoding
# matricies because it can compute the sine integral relatively
# quickly:
from scipy.special import sici

from bionet.utils.numpy_extras import mdot 

# Pseudoinverse singular value cutoff:
__pinv_rcond__ = 1e-8

def asdm_recoverable_strict(u, bw, b, d, k):
    """Determine whether a time-encoded signal can be perfectly
    recovered using an ASDM decoder with the specified parameters.

    Parameters
    ----------
    u: numpy array
        Signal to test.
    bw: float
        Signal bandwidth (in rad/s).
    b: float
        Decoder bias.
    d: float
        Decoder threshold.
    k: float
        Decoder integration constant.

    Raises
    ------
    ValueError
        When the signal cannot be perfectly recovered.
    """
    
    c = max(abs(u))
    if c >= b:
        raise ValueError('bias too low')
    elif 2*k*d/(b-c)*bw/pi >= 1.0:
        raise ValueError('reconstruction condition not satisfied;'+
                         'try raising b or reducing k or d')
    else:
        return True

def asdm_recoverable(u, bw, b, d, k):
    """Determine whether a time-encoded signal can be perfectly
    recovered using an ASDM decoder with the specified parameters.

    Parameters
    ----------
    u: numpy array
        Signal to test.
    bw: float
        Signal bandwidth (in rad/s).
    b: float
        Decoder bias.
    d: float
        Decoder threshold.
    k: float
        Decoder integration constant.

    Raises
    ------
    ValueError
        When the signal cannot be perfectly recovered.
        
    Note
    ----
    The bound assumed by this check is not as strict as that described in
    most of Prof. Lazar's papers.
    """
    
    c = max(abs(u))
    if c >= b:
        raise ValueError('bias too low')
    elif (2*k*d/b)*bw/pi >= 1.0:
        raise ValueError('reconstruction condition not satisfied;'+
                         'try raising b or reducing k or d')
    else:
        return True

def asdm_encode(u, dt, b, d, k=1.0, dte=0.0, y=0.0, interval=0.0,
                sgn=1, quad_method='trapz', full_output=False):
    """Encode a finite length signal using an asynchronous sigma-delta
    modulator.

    Parameters
    ----------
    u: numpy array of floats
        Signal to encode.
    dt: float
        Sampling resolution of input signal; the sampling frequency
        is 1/dt Hz.
    b: float
        Encoder bias.
    d: float
        Encoder threshold.
    k: float
        Encoder integration constant.
    dte: float
        Sampling resolution assumed by the encoder (s).
        This may not exceed dt.
    y: float 
        Initial value of integrator.
    interval: float
        Time since last spike (in s).
    sgn: {+1, -1}
        Sign of integrator.
    quad_method: {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal).
    full_output: boolean
        If set, the function returns new values for y, interval, and
        sgn.

    Notes
    -----
    When trapezoidal integration is used, the value of the integral
    will not be computed for the very last entry in u.
    
    References
    ----------
    Lazar, A.A., and L.T. Toth. Perfect Recovery and Sensitivity Analysis
    of Time Encoded Bandlimited Signals. IEEE Transactions on Circuits
    and Systems-I: Regular Papers, 51(10):2060-2073, October 2004.
    """
    
    nu = len(u)
    if nu == 0:        
        if full_output:
            return array((),float),y,interval,sgn
        else:
            return array((),float)

    # Check whether the encoding resolution is finer than that of the
    # original sampled signal:
    if dte > dt:
        raise ValueError('encoding time resolution must not exceeed original signal resolution')
    if dte < 0:
        raise ValueError('encoding time resolution must be nonnegative')
    if dte != 0 and dte != dt:
        u = resample(u,len(u)*int(dt/dte))
        nu = len(u)
        dt = dte
        
    # Use a list rather than an array to save the spike intervals
    # because the number of spikes is not fixed:
    s = []

    # Choose integration method and set the number of points over
    # which to integrate the input (see note above). This allows the
    # use of one loop below to perform the integration regardless of
    # the method chosen:
    if quad_method == 'rect':
        compute_y = lambda y,sgn,i: y + dt*(sgn*b+u[i])/k
        last = nu
    elif quad_method == 'trapz':
        compute_y = lambda y,sgn,i: y + dt*(sgn*b+(u[i]+u[i+1])/2.0)/k
        last = nu-1
    else:
        raise ValueError('unrecognized quadrature method')
    
    for i in xrange(last):
        y = compute_y(y,sgn,i)
        interval += dt
        if abs(y) >= d:
            s.append(interval)
            interval = 0.0
            y = d*sgn
            sgn = -sgn

    if full_output:
        return array(s),y,interval,sgn
    else:
        return array(s)

def asdm_decode(s, dur, dt, bw, b, d, k):    
    """Decode a finite length signal encoded by an asynchronous sigma-delta
    modulator.

    Parameters
    ----------
    s: numpy array of floats
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
        Encoder integrator constant.
        
    References
    ----------
    Lazar, A.A., and L.T. Toth. Perfect Recovery and Sensitivity Analysis
    of Time Encoded Bandlimited Signals. IEEE Transactions on Circuits
    and Systems-I: Regular Papers, 51(10):2060-2073, October 2004.
    """

    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements')
    
    ts = cumsum(s)
    tsh = (ts[0:-1]+ts[1:])/2
    nsh = len(tsh)
    
    nt = int(dur/dt)
    t = linspace(0,dur,nt)
    
    bwpi = bw/pi
    
    # Compute G matrix:
    G = empty((nsh,nsh),float)
    for j in xrange(nsh):

        # Compute the values for all of the sincs so that they do not
        # need to each be recomputed when determining the integrals
        # between spike times:
        temp = sici(bw*(ts-tsh[j]))[0]/pi
        for i in xrange(nsh):
            G[i,j] = temp[i+1]-temp[i]
    G_inv = pinv(G,__pinv_rcond__)

    # Compute quanta:
    q = array([(-1)**i for i in xrange(1,nsh+1)])*(2*k*d-b*s[1:])

    # Reconstruct signal by adding up the weighted sinc functions. The
    # weighted sinc functions are computed on the fly here to save
    # memory:
    u_rec = zeros(nt,float)
    c = dot(G_inv,q)
    for i in xrange(nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def asdm_decode_ins(s, dur, dt, bw, b):    
    """Decode a finite length signal encoded by an asynchronous sigma-delta
    modulator using a threshold-insensitive recovery algorithm.

    Parameters
    ----------
    s: numpy array of floats
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

    References
    ----------
    Lazar, A.A., and L.T. Toth. Perfect Recovery and Sensitivity Analysis
    of Time Encoded Bandlimited Signals. IEEE Transactions on Circuits
    and Systems-I: Regular Papers, 51(10):2060-2073, October 2004.

    Notes
    -----
    This implementation of the decoding algorithm has a slightly better
    recovery error than the implementations in asdm_decode_ins2()
    and asdm_decode_ins3().
    """
    
    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements') 
    
    ts = cumsum(s)
    tsh = (ts[0:-1]+ts[1:])/2
    nsh = len(tsh)
    
    nt = int(dur/dt)
    t = linspace(0,dur,nt)
    
    bwpi = bw/pi
    
    # Compute G matrix:
    G = empty((nsh,nsh),float)
    for j in xrange(nsh):

        # Compute the values for all of the sinc functions so that
        # they do not need to each be recomputed when determining the
        # integrals between spike times:
        temp = sici(bw*(ts-tsh[j]))[0]/pi
        for i in xrange(nsh):
            G[i,j] = temp[i+1]-temp[i]
    
    # Apply compensation principle:
    B = diag(ones(nsh-1),-1)+eye(nsh)
    Bq = array([(-1)**i for i in xrange(nsh)])*b*(s[1:]-s[:-1])

    # Reconstruct signal by adding up the weighted sinc functions; the
    # first row of B is removed to eliminate boundary issues. The
    # weighted sinc functions are computed on the fly to save memory:
    u_rec = zeros(nt,float)
    c = dot(pinv(dot(B[1:,:],G),__pinv_rcond__),Bq[1:,newaxis])
    for i in xrange(nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def asdm_decode_ins2(s, dur, dt, bw, b):    
    """Decode a finite length signal encoded by an asynchronous sigma-delta
    modulator using a threshold-insensitive recovery algorithm.

    Parameters
    ----------
    s: numpy array of floats
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

    References
    ----------
    Lazar, A.A., and L.T. Toth. Perfect Recovery and Sensitivity Analysis
    of Time Encoded Bandlimited Signals. IEEE Transactions on Circuits
    and Systems-I: Regular Papers, 51(10):2060-2073, October 2004.
    """
    
    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements') 
    
    ts = cumsum(s)
    tsh = (ts[0:-1]+ts[1:])/2
    nsh = len(tsh)
    
    nt = int(dur/dt)
    t = linspace(0,dur,nt)
    
    bwpi = bw/pi
    
    # Compute G matrix:
    G = empty((nsh,nsh),float)
    for j in xrange(nsh):

        # Compute the values for all of the sinc functions so that
        # they do not need to each be recomputed when determining the
        # integrals between spike times:
        temp = sici(bw*(ts-tsh[j]))[0]/pi
        for i in xrange(nsh):
            G[i,j] = temp[i+1]-temp[i]

    # Apply compensation principle:    
    B = diag(ones(nsh-1),-1)+eye(nsh)
    B_inv = inv(B)
    Bq = array([(-1)**i for i in xrange(nsh)])*hstack((0,b*diff(s[1:])))

    # Blank the nonzero entries in the first row of B and the last column
    # of B_inv to eliminate boundary issues:
    B[0,0] = B_inv[-1,-1] = 0.0

    # Reconstruct signal by adding up the weighted sinc functions. The
    # weighted sinc functions are computed on the fly to save memory:
    u_rec = zeros(nt,float)
    c = mdot(B_inv,pinv(mdot(B,G,B_inv),__pinv_rcond__),Bq[:,newaxis])
    for i in xrange(nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def asdm_decode_ins3(s, dur, dt, bw, b):    
    """Decode a finite length signal encoded by an asynchronous sigma-delta
    modulator using a threshold-insensitive recovery algorithm.

    Parameters
    ----------
    s: numpy array of floats
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

    References
    ----------
    Lazar, A.A., and L.T. Toth. Perfect Recovery and Sensitivity Analysis
    of Time Encoded Bandlimited Signals. IEEE Transactions on Circuits
    and Systems-I: Regular Papers, 51(10):2060-2073, October 2004.

    Notes
    -----
    This implementation of the decoding algorithm is slightly faster than that
    in asdm_decode_ins() and asdm_decode_ins2(), although its recovery error is
    not as good as that of asdm_decode_ins().
    """
    
    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements') 
    
    ts = cumsum(s)
    tsh = (ts[0:-1]+ts[1:])/2
    nsh = len(tsh)-1
    
    nt = int(dur/dt)
    t = linspace(0,dur,nt)
    
    bwpi = bw/pi
    
    # Compute G matrix:
    G = empty((nsh,nsh),float)
    for j in xrange(nsh):

        # Compute the values for all of the sinc functions so that
        # they do not need to each be recomputed when determining the
        # integrals between spike times:
        temp = sici(bw*(ts-tsh[j]))[0]/pi
        for i in xrange(nsh):
            G[i,j] = temp[i+2]-temp[i]
    
    # Apply compensation principle:
    Bq = array([(-1)**i for i in xrange(1,nsh+1)])*b*(s[2:]-s[1:-1])

    # Reconstruct signal by adding up the weighted sinc functions. The
    # weighted sinc functions are computed on the fly to save memory:
    u_rec = zeros(nt,float)
    c = dot(pinv(G,__pinv_rcond__),Bq[:,newaxis])
    for i in xrange(nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

