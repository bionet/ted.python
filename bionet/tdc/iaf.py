#!/usr/bin/env python

"""
Time encoding and decoding algorithms that make use of the
integrate-and-fire neuron model.
"""

__all__ = ['iaf_recoverable','iaf_encode','iaf_decode',
           'iaf_decode_ins','iaf_decode_fast','iaf_decode_rec']

from numpy import array, abs, max, log, pi, real, imag, isreal, float,\
     isinf, exp, nonzero, diff, hstack, arange, triu, diag, dot, inf,\
     ones, zeros, sinc, ravel, newaxis, eye, empty, shape, conjugate,\
     linspace, cumsum
from numpy.linalg import inv, pinv
from scipy.integrate import quad
from scipy.signal import resample

# The sici() function is used to obtain the values in the decoding
# matricies because it can compute the sine integral relatively
# quickly:
from scipy.special import sici

from bionet.utils.numpy_extras import mdot

def iaf_recoverable(u, bw, b, d, R, C):
    """Determine whether a time-encoded signal can be perfectly
    recovered using an IAF decoder with the specified parameters.

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
    R: float
        Neuron resistance.
    C: float
        Neuron capacitance.

    Raises
    ------
    ValueError
        When the signal cannot be perfectly recovered.
    """

    c = max(abs(u))
    if c >= b:
        raise ValueError('bias too low')
    r = R*C*log(1-d/(d-(b-c)*R))*bw/pi
    e = d/((b-c)*R)
    if not isreal(r):
        #print 'r = %f + %fj' % real(r), imag(r)
        raise ValueError('reconstruction condition not satisfied')
    elif r >= (1-e)/(1+e):
        #print 'r = %f, (1-e)/(1+e) = %f' % (r, (1-e)/(1+e))
        raise ValueError('reconstruction condition not satisfied;'+
                         'try raising b or reducing d')
    else:
        return True

def iaf_encode(u, dt, b, d, R=inf, C=1.0, dte=0, y=0.0, interval=0.0,
               quad_method='trapz', full_output=False):
    """Encode a finite length signal using an integrate-and-fire
    neuron.

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
    R: float
        Neuron resistance.
    C: float
        Neuron capacitance.
    dte: float
        Sampling resolution assumed by the encoder (s).
        This may not exceed dt.
    y: float
        Initial value of integrator.
    interval: float
        Time since last spike (in s).
    quad_method: {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal) when the
        neuron is not leaky; exponential Euler integration is used
        when the neuron is leaky.
    full_output: boolean
        If set, the function returns new values for y and interval.

    Notes
    -----
    When trapezoidal integration is used, the value of the integral
    will not be computed for the very last entry in u.

    References
    ----------
    Lazar, A.A. Unpublished lecture notes.
    """

    nu = len(u)
    if nu == 0:
        if full_output:
            return array((),float),array((),float)
        else:
            return array((),float)
    
    # Check whether the encoding resolution is finer than that of the
    # original sampled signal:
    if dte > dt:
        raise ValueError('encoding time resolution must not exceeed original signal resolution')
    if dte < 0:
        raise ValueError('encoding time resolution must be nonnegative')
    if dte != 0:        
        u = resample(u,len(u)*int(dt/dte))
        dt = dte

    # Use a list rather than an array to save the spike intervals
    # because the number of spikes is not fixed:
    s = []

    # Choose integration method:
    if isinf(R):        
        y += dt*(u[0]+b)/C

        if quad_method == 'rect':
            compute_y = lambda y_curr, *args: y_curr + dt*(b+args[0])/C
        elif quad_method == 'trapz':
            compute_y = lambda y_curr, *args: y_curr + dt*(b+(args[0]+args[1])/2.0)/C
        else:
            raise ValueError('unrecognized quadrature method')
    else:

        # When the neuron is leaky, use the exponential Euler method to perform
        # the encoding:
        RC = R*C
        y += dt*(u[0]+b)*exp(-dt/RC)
        compute_y = lambda y_curr, *args: \
            y_curr*exp(-dt/RC)+R*(1-exp(-dt/RC))*(b+args[0])

    # The interval between spikes is saved between iterations rather than the
    # absolute time so as to avoid overflow problems for very long signals:
    for i in xrange(1,nu):
        y = compute_y(y,u[i],u[i-1])
        interval += dt
        if y >= d:
            s.append(interval)
            interval = 0.0
            y = 0.0

    if full_output:
        return array(s),y,interval
    else:
        return array(s)

def iaf_decode(s, dur, dt, bw, b, d, R=inf, C=1.0):
    """Decode a finite length signal encoded by an integrate-and-fire
    neuron.

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
    R: float
        Neuron resistance.
    C: float
        Neuron capacitance.
        
    References
    ----------
    Lazar, A.A. Unpublished lecture notes.
    """

    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements')
    
    ts = cumsum(s) 
    tsh = (ts[0:-1]+ts[1:])/2

    nt = int(dur/dt)
    t = linspace(0,dur,nt)
    nsh = len(tsh)

    bwpi = bw/pi
    RC = R*C

    # Compute G matrix and quanta:
    G = empty((nsh,nsh),float)
    if isinf(R):
        for i in xrange(nsh):
            for j in xrange(nsh):
                G[i,j] = (sici(bw*(ts[i+1]-tsh[j]))[0]- \
                          sici(bw*(ts[i]-tsh[j]))[0])/pi
        q = C*d-b*s[1:]
    else:
        for i in xrange(nsh):
            for j in xrange(nsh):

                # XXX: This explicit integration should replaced with
                # a more efficient expression, e.g., possibly using
                # the exponential integral:
                f = lambda t:sinc(bwpi*(t-tsh[j]))*bwpi*exp((ts[i+1]-t)/-RC)
                G[i,j] = quad(f,ts[i],ts[i+1])[0]
        q = C*(d+b*R*(exp(-s[1:]/RC)-1))
        
    G_inv = pinv(G)

    # Reconstruct signal by adding up the weighted sinc functions.
    u_rec = zeros(nt,float)
    c = dot(G_inv,q)
    for i in xrange(nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def iaf_decode_ins(s, dur, dt, bw, b, R=inf, C=1.0):
    """Decode a finite length signal encoded by an integrate-and-fire
    neuron using a threshold-insensitive recovery algorithm.

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
    R: float
        Neuron resistance.
    C: float
        Neuron capacitance.
        
    References
    ----------
    Lazar, A.A. Unpublished lecture notes.
    """

    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements')
    
    ts = cumsum(s) 
    tsh = (ts[0:-1]+ts[1:])/2

    nt = int(dur/dt)
    t = linspace(0,dur,nt)
    nsh = len(tsh)

    bwpi = bw/pi
    RC = R*C
    
    # Compute G matrix:
    G = empty((nsh,nsh),float)
    if isinf(R):
        for i in xrange(nsh):
            for j in xrange(nsh):
                G[i,j] = (sici(bw*(ts[i+1]-tsh[j]))[0]- \
                          sici(bw*(ts[i]-tsh[j]))[0])/pi
    else:
        for i in xrange(nsh):
            for j in xrange(nsh):

                # XXX: This explicit integration should replaced with
                # a more efficient expression, e.g., possibly using
                # the exponential integral:
                f = lambda t:sinc(bwpi*(t-tsh[j]))*bwpi*exp((ts[i+1]-t)/-RC)
                G[i,j] = quad(f,ts[i],ts[i+1])[0]
        
    G_inv = pinv(G)

    # Apply compensation principle:
    B = diag(ones(nsh-1),1)-eye(nsh)
    B_inv = inv(B)
    if isinf(R):
        Bq = -b*diff(s)
    else:
        Bq = RC*b*diff(exp(-s/RC))

    # Blank the last rows and columns of B and
    # B_inv to eliminate boundary issues:
    B[:,-1] = B[-1,:] = 0
    B_inv[:,-1] = B_inv[-1,:] = 0
    
    # Reconstruct signal by adding up the weighted sinc functions:
    u_rec = zeros(nt,float)
    c = mdot(B_inv,pinv(mdot(B,G,B_inv)),Bq[:,newaxis])
    for i in xrange(nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def iaf_decode_fast(s, dur, dt, bw, M, b, d, R=inf, C=1.0):
    """Decode a finite length signal encoded by an integrate-and-fire
    neuron using a threshold-insensitive recovery algorithm.

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
    M: int
        Number of bins used by the fast algorithm.
    b: float
        Encoder bias.
    d: float
        Encoder threshold.
    R: float
        Neuron resistance.
    C: float
        Neuron capacitance.
        
    References
    ----------
    Lazar, A.A, E.K. Simonyi, and L.T. Toth. Fast Recovery Algoritms
    for Time Encoded Bandlimited Signals. IEEE Transactions on
    Circuits and Systems-I: Regular Papers, 51:237-240, March 2005.
    """
    
    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements')
    
    # Convert M in the event that an integer was specified:
    M = float(M)

    ts = cumsum(s) 
    tsh = (ts[0:-1]+ts[1:])/2

    nt = int(dur/dt)
    t = linspace(0,dur,nt)
    nsh = len(tsh)

    RC = R*C
    jbwM = 1j*bw/M

    # Compute quanta:
    if isinf(R):
        q = C*d-b*s[1:]
    else:
        q = C*(d+b*R*(exp(-s[1:]/RC)-1))

    # Compute approximation coefficients:
    a = bw/(pi*(2*M+1))
    m = arange(-M,M+1)
    P_inv = -triu(ones((nsh,nsh)))
    S = exp(-jbwM*dot(m[:,newaxis],ts[:-1][newaxis]))
    D = diag(s[1:])
    SD = dot(S,D)
    T = mdot(a,SD,conjugate(S.T))
    dd = mdot(a,pinv(T),SD,P_inv,q[:,newaxis])

    # Reconstruct signal:
    return ravel(real(jbwM*dot(m*dd.T,exp(jbwM*m[:,newaxis]*t))))

def iaf_decode_rec(s, dur, dt, bw, L, b, d, R=inf, C=1.0):
    """Decode a finite length signal encoded by an integrate-and-fire
    neuron using a recursive decoding operator algorithm.

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
    L: int
        Number of times to recursively apply the reconstruction operator.
    b: float
        Encoder bias.
    d: float
        Encoder threshold.
    R: float
        Neuron resistance.
    C: float
        Neuron capacitance.
        
    References
    ----------
    Lazar, A.A. Unpublished lecture notes.

    Notes
    -----
    The implementation of this algorithm can potentially be memory-intensive.
    """

    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements')
    
    ts = cumsum(s) 
    tsh = (ts[0:-1]+ts[1:])/2

    nt = int(dur/dt)
    t = linspace(0,dur,nt)
    nsh = len(tsh)

    bwpi = bw/pi
    RC = R*C
    
    # Compute shifted sincs, G matrix, and quanta:
    g = empty((nt,nsh),float)
    G = empty((nsh,nsh),float)
    if isinf(R):
        for i in xrange(nsh):
            g[:,i] = sinc(bwpi*(t-ts[i]))*bwpi
            for j in xrange(nsh):
                G[i,j] = (sici(bw*(ts[i+1]-tsh[j]))[0]- \
                          sici(bw*(ts[i]-tsh[j]))[0])/pi
        q = C*d-b*s[1:]
    else:
        for i in xrange(nsh):
            g[:,i] = sinc(bwpi*(t-ts[i]))*bwpi
            for j in xrange(nsh):

                # XXX: This explicit integration should replaced with
                # a more efficient expression, e.g., possibly using
                # the exponential integral:
                f = lambda t:sinc(bwpi*(t-tsh[j]))*bwpi*exp((ts[i+1]-t)/-RC)
                G[i,j] = quad(f,ts[i],ts[i+1])[0]
        q = C*(d+b*R*(exp(-s[1:]/RC)-1))

    IG = eye(nsh)-G

    # Recursively reconstruct signal:
    u_rec = empty(nt,float)
    IGj = eye(nsh)
    for j in xrange(L+1):
        u_curr = ravel(mdot(g,IGj,q[:,newaxis]))
        if j>0:
            u_rec += u_curr
        else:
            u_rec = u_curr
        IGj = dot(IGj,IG)

    return u_rec
