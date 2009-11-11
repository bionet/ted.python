#!/usr/bin/env python

"""
Time encoding and decoding algorithms that make use of the
asynchronous sigma-delta modulator.
"""

__all__ = ['asdm_recoverable', 'asdm_encode', 'asdm_decode',
           'asdm_decode_ins', 'asdm_decode_fast',
	   'asdm_decode_pop', 'asdm_decode_pop_ins']
           
from numpy import abs, arange, array, conjugate, cumsum, diag, dot, \
        empty, eye, exp, float, max, newaxis, ones, pi, ravel, real, \
        sinc, triu, zeros
from numpy.linalg import pinv
from scipy.signal import resample

# The sici() function is used to construct the decoding matrix G
# because it can compute the sine integral relatively quickly:
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
        If set, the function returns the encoded data block followed
        by the given parameters (with updated values for y, interval, and
        sgn). This is useful when the function is called repeatedly to
        encode a long signal.

    Notes
    -----
    When trapezoidal integration is used, the value of the integral
    will not be computed for the very last entry in u.
    
    """
    
    nu = len(u)
    if nu == 0:        
        if full_output:
            return array((), float), dt, b, d, k, dte, y, interval, sgn, \
               quad_method, full_output
        else:
            return array((), float)

    # Check whether the encoding resolution is finer than that of the
    # original sampled signal:
    if dte > dt:
        raise ValueError('encoding time resolution must not exceeed original signal resolution')
    if dte < 0:
        raise ValueError('encoding time resolution must be nonnegative')
    if dte != 0 and dte != dt:
        u = resample(u, len(u)*int(dt/dte))
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
        compute_y = lambda y, sgn, i: y + dt*(sgn*b+u[i])/k
        last = nu
    elif quad_method == 'trapz':
        compute_y = lambda y, sgn, i : y + dt*(sgn*b+(u[i]+u[i+1])/2.0)/k
        last = nu-1
    else:
        raise ValueError('unrecognized quadrature method')
    
    for i in xrange(last):
        y = compute_y(y, sgn, i)
        interval += dt
        if abs(y) >= d:
            s.append(interval)
            interval = 0.0
            y = d*sgn
            sgn = -sgn

    if full_output:
        return array(s), dt, b, d, k, dte, y, interval, sgn, \
               quad_method, full_output
    else:
        return array(s)

def asdm_decode(s, dur, dt, bw, b, d, k=1.0, sgn=-1):    
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
    sgn: {-1,1}
        Sign of first spike.
        
    """

    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements')
    
    ts = cumsum(s)
    tsh = (ts[0:-1]+ts[1:])/2
    nsh = len(tsh)
    
    t = arange(0, dur, dt)
    
    bwpi = bw/pi
    
    # Compute G matrix:
    G = empty((nsh, nsh), float)
    for j in xrange(nsh):

        # Compute the values for all of the sincs so that they do not
        # need to each be recomputed when determining the integrals
        # between spike times:
        temp = sici(bw*(ts-tsh[j]))[0]/pi
        for i in xrange(nsh):
            G[i, j] = temp[i+1]-temp[i]
    G_inv = pinv(G, __pinv_rcond__)

    # Compute quanta:
    if sgn == -1:
        q = array([(-1)**i for i in xrange(1, nsh+1)])*(2*k*d-b*s[1:])
    else:
        q = array([(-1)**i for i in xrange(0, nsh)])*(2*k*d-b*s[1:])
        
    # Reconstruct signal by adding up the weighted sinc functions. The
    # weighted sinc functions are computed on the fly here to save
    # memory:
    u_rec = zeros(len(t), float)
    c = dot(G_inv, q)
    for i in xrange(nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def asdm_decode_ins(s, dur, dt, bw, b, sgn=-1):    
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
    sgn: {-1,1}
        Sign of first spike.

    """
    
    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements') 
    
    ts = cumsum(s)
    tsh = (ts[0:-1]+ts[1:])/2
    nsh = len(tsh)
    
    t = arange(0, dur, dt)
    
    bwpi = bw/pi
    
    # Compute G matrix:
    G = empty((nsh, nsh), float)
    for j in xrange(nsh):

        # Compute the values for all of the sinc functions so that
        # they do not need to each be recomputed when determining the
        # integrals between spike times:
        temp = sici(bw*(ts-tsh[j]))[0]/pi
        for i in xrange(nsh):
            G[i, j] = temp[i+1]-temp[i]
    
    # Apply compensation principle:
    B = diag(ones(nsh-1), -1)+eye(nsh)
    if sgn == -1:
        Bq = array([(-1)**i for i in xrange(nsh)])*b*(s[1:]-s[:-1])
    else:
        Bq = array([(-1)**i for i in xrange(1, nsh+1)])*b*(s[1:]-s[:-1])
        
    # Reconstruct signal by adding up the weighted sinc functions; the
    # first row of B is removed to eliminate boundary issues. The
    # weighted sinc functions are computed on the fly to save memory:
    u_rec = zeros(len(t), float)
    c = dot(pinv(dot(B[1:, :], G), __pinv_rcond__), Bq[1:, newaxis])
    for i in xrange(nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def asdm_decode_fast(s, dur, dt, bw, M, b, d, k=1.0, sgn=-1):
    """Decode a finite length signal encoded by an asynchronous sigma-delta
    modulator using a fast recovery algorithm.

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
    k: float
        Encoder integrator constant.
    sgn: {-1,1}
        Sign of first spike.
        
    """
    
    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements')
    
    # Convert M in the event that an integer was specified:
    M = float(M)

    ts = cumsum(s) 
    tsh = (ts[0:-1]+ts[1:])/2
    nsh = len(tsh)
    
    t = arange(0, dur, dt)
    
    jbwM = 1j*bw/M

    # Compute quanta:
    if sgn == -1:
        q = array([(-1)**i for i in xrange(1, nsh+1)])*(2*k*d-b*s[1:])
    else:
        q = array([(-1)**i for i in xrange(0, nsh)])*(2*k*d-b*s[1:])
        
    # Compute approximation coefficients:
    a = bw/(pi*(2*M+1))
    m = arange(-M, M+1)
    P_inv = -triu(ones((nsh, nsh)))
    S = exp(-jbwM*dot(m[:, newaxis], ts[:-1][newaxis]))
    D = diag(s[1:])
    SD = dot(S, D)
    T = mdot(a, SD, conjugate(S.T))
    dd = mdot(a, pinv(T, __pinv_rcond__), SD, P_inv, q[:, newaxis])

    # Reconstruct signal:
    return ravel(real(jbwM*dot(m*dd.T, exp(jbwM*m[:, newaxis]*t))))

def asdm_decode_pop(s_list, dur, dt, bw, b_list, d_list, k_list, sgn_list=[]):
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
    sgn_list: list of integers {-1, 1}
        List of signs of first spikes in trains.
    
    Notes
    -----
    The number of spikes contributed by each neuron may differ from the
    number contributed by other neurons.

    """

    M = len(s_list)
    if not M:
        raise ValueError('no spike data given')

    # Set sign of first spikes:
    if sgn_list == []:
        sgn_list = M*[-1]
    if len(sgn_list) != M:
        raise ValueError('incorrect number of first spike signs')

    bwpi = bw/pi
    
    # Compute the midpoints between spikes:
    ts_list = map(cumsum, s_list)
    tsh_list = map(lambda ts:(ts[0:-1]+ts[1:])/2, ts_list)

    # Compute number of spikes in each spike list:
    Ns_list = map(len, ts_list)
    Nsh_list = map(len, tsh_list)
        
    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    Nsh_sum = sum(Nsh_list)
    G = empty((Nsh_sum, Nsh_sum), float)
    q = empty((Nsh_sum, 1), float)
    for l in xrange(M):
        for m in xrange(M):
            G_block = empty((Nsh_list[l], Nsh_list[m]), float)

            # Compute the values for all of the sincs so that they
            # do not need to each be recomputed when determining
            # the integrals between spike times:
            for k in xrange(Nsh_list[m]):
                temp = sici(bw*(ts_list[l]-tsh_list[m][k]))[0]/pi
                for n in xrange(Nsh_list[l]):
                    G_block[n, k] = temp[n+1]-temp[n]

            G[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]),
              sum(Nsh_list[:m]):sum(Nsh_list[:m+1])] = G_block

        # Compute the quanta:
        if sgn_list[l] == -1:
            q[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]), 0] = \
                array([(-1)**i for i in xrange(1, Nsh_list[l]+1)])* \
                (2*k_list[l]*d_list[l]-b_list[l]*s_list[l][1:])
        else:
            q[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]), 0] = \
                array([(-1)**i for i in xrange(0, Nsh_list[l])])* \
                (2*k_list[l]*d_list[l]-b_list[l]*s_list[l][1:])
            
    # Compute the reconstruction coefficients:
    c = dot(pinv(G), q)

    # Reconstruct the signal using the coefficients:
    t = arange(0, dur, dt)
    u_rec = zeros(len(t), float)
    for m in xrange(M):
        for k in xrange(Nsh_list[m]):
            u_rec += sinc(bwpi*(t-tsh_list[m][k]))*bwpi*c[sum(Nsh_list[:m])+k, 0]
    return u_rec

def asdm_decode_pop_ins(s_list, dur, dt, bw, b_list, sgn_list=[]):
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

    # Set sign of first spikes:
    if sgn_list == []:
        sgn_list = M*[-1]
    if len(sgn_list) != M:
        raise ValueError('incorrect number of first spike signs')

    bwpi = bw/pi
    
    # Compute the midpoints between spikes:
    ts_list = map(cumsum, s_list)
    tsh_list = map(lambda ts:(ts[0:-1]+ts[1:])/2, ts_list)

    # Compute number of spikes in each spike list:
    Nsh_list = map(lambda x: len(x)-1, tsh_list)
    
    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    Nsh_sum = sum(Nsh_list)
    G = empty((Nsh_sum, Nsh_sum), float)
    Bq = empty((Nsh_sum, 1), float)
    for l in xrange(M):
        for m in xrange(M):
            G_block = empty((Nsh_list[l], Nsh_list[m]), float)

            # Compute the values for all of the sincs so that they
            # do not need to each be recomputed when determining
            # the integrals between spike times:
            for k in xrange(Nsh_list[m]):
                temp = sici(bw*(ts_list[l]-tsh_list[m][k]))[0]/pi
                for n in xrange(Nsh_list[l]):
                    G_block[n, k] = temp[n+2]-temp[n]

            G[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]),
              sum(Nsh_list[:m]):sum(Nsh_list[:m+1])] = G_block

        # Compute the quanta:
        if sgn_list[l] == -1:
            Bq[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]), 0] = \
                array([(-1)**i for i in xrange(1, Nsh_list[l]+1)])* \
                b_list[l]*(s_list[l][2:]-s_list[l][1:-1])
        else:
            Bq[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]), 0] = \
                array([(-1)**i for i in xrange(0, Nsh_list[l])])* \
                b_list[l]*(s_list[l][2:]-s_list[l][1:-1])

    # Compute the reconstruction coefficients:
    c = dot(pinv(G), Bq)

    # Reconstruct the signal using the coefficients:
    t = arange(0, dur, dt)
    u_rec = zeros(len(t), float)
    for m in xrange(M):
        for k in xrange(Nsh_list[m]):
            u_rec += sinc(bwpi*(t-tsh_list[m][k]))*bwpi*c[sum(Nsh_list[:m])+k, 0]
    return u_rec

