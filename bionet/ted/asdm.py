#!/usr/bin/env python

"""
Time encoding and decoding algorithms that make use of the
asynchronous sigma-delta modulator.
 
- asdm_decode         - Decode a signal encoded by an ASDM encoder.
- asdm_decode_fast    - Fast ASDM decoding algorithm.
- asdm_decode_ins     - Parameter-insensitive version of asdm_decode.
- asdm_decode_pop     - asdm_decode for a population of ASDM encoders.
- asdm_decode_pop_ins - Parameter-insensitive version of asdm_decode_pop.
- asdm_encode         - Encode a signal using an ASDM encoder.
- asdm_recoverable    - Check encoder parameters for decoding feasibility.

"""

__all__ = ['asdm_recoverable', 'asdm_encode', 'asdm_decode',
           'asdm_decode_ins', 'asdm_decode_fast',
	   'asdm_decode_pop', 'asdm_decode_pop_ins']
           
from numpy import abs, arange, array, asarray, conjugate, cumsum, \
        diag, dot, empty, eye, exp, float, max, newaxis, ones, pi, \
        ravel, real, sinc, triu, zeros
from numpy.linalg import pinv
from scipy.signal import resample

# The sici() function is used to construct the decoding matrix G
# because it can compute the sine integral relatively quickly:
from scipy.special import sici
from bionet.utils.numpy_extras import mdot 

from bionet.ted.vtdm import asdm_decode_vander, \
     asdm_decode_vander_ins

__all__ += ['asdm_decode_vander', 'asdm_decode_vander_ins']

# Pseudoinverse singular value cutoff:
__pinv_rcond__ = 1e-8

def asdm_recoverable_strict(u, bw, b, d, k):
    """
    Determine whether a signal can be perfectly recovered with an ASDM
    decoder with the specified parameters.
    
    Parameters
    ----------
    u : array_like of floats
        Signal to test.
    bw : float
        Signal bandwidth (in rad/s).
    b : float
        Decoder bias.
    d : float
        Decoder threshold.
    k : float
        Decoder integration constant.

    Returns
    -------
    rec : bool
        True if the specified signal is recoverable.
        
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
    """
    Determine whether a signal can be perfectly recovered with an ASDM
    decoder with the specified parameters.

    Parameters
    ----------
    u: array_like of floats
        Signal to test.
    bw: float
        Signal bandwidth (in rad/s).
    b: float
        Decoder bias.
    d: float
        Decoder threshold.
    k: float
        Decoder integration constant.

    Returns
    -------
    rec : bool
        True if the specified signal is recoverable.

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
    """
    Encode a finite length signal with an ASDM encoder.

    Parameters
    ----------
    u : array_like of floats
        Signal to encode.
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is 1/dt Hz.
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    k : float
        Encoder integration constant.
    dte : float
        Sampling resolution assumed by the encoder (s).
        This may not exceed `dt`.
    y : float 
        Initial value of integrator.
    interval : float
        Time since last spike (in s).
    sgn : {+1, -1}
        Sign of integrator.
    quad_method : {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal).
    full_output : bool
        If set, the function returns the encoded data block followed
        by the given parameters (with updated values for `y`, `interval`, and
        `sgn`). This is useful when the function is called repeatedly to
        encode a long signal.

    Returns
    -------
    s : ndarray of floats
        If `full_output` == False, returns the signal encoded as an
        array of time intervals between spikes.
    s, dt, b, d, k, dte, y, interval, sgn, quad_method, full_output : tuple
        If `full_output` == True, returns the encoded signal
        followed by updated encoder parameters.

    Notes
    -----
    When trapezoidal integration is used, the value of the integral
    will not be computed for the very last entry in `u`.
    
    """
    
    Nu = len(u)
    if Nu == 0:        
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
        Nu = len(u)
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
        last = Nu
    elif quad_method == 'trapz':
        compute_y = lambda y, sgn, i : y + dt*(sgn*b+(u[i]+u[i+1])/2.0)/k
        last = Nu-1
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
    """
    Decode a signal encoded with an ASDM encoder.

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
    k : float
        Encoder integrator constant.
    sgn : {-1, 1}
        Sign of first spike.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.

    """

    Ns = len(s)
    if Ns < 2:
        raise ValueError('s must contain at least 2 elements')

    # Cast s to an ndarray to permit ndarray operations:
    s = asarray(s)

    # Compute the spike times:
    ts = cumsum(s)

    # Compute the midpoints between spike times:
    tsh = (ts[0:-1]+ts[1:])/2
    Nsh = len(tsh)
    
    bwpi = bw/pi
    
    # Compute G matrix:
    G = empty((Nsh, Nsh), float)
    for j in xrange(Nsh):

        # Compute the values for all of the sincs so that they do not
        # need to each be recomputed when determining the integrals
        # between spike times:
        temp = sici(bw*(ts-tsh[j]))[0]/pi
        for i in xrange(Nsh):
            G[i, j] = temp[i+1]-temp[i]
    G_inv = pinv(G, __pinv_rcond__)

    # Compute quanta:
    if sgn == -1:
        q = array([(-1)**i for i in xrange(1, Nsh+1)])*(2*k*d-b*s[1:])
    else:
        q = array([(-1)**i for i in xrange(0, Nsh)])*(2*k*d-b*s[1:])
        
    # Reconstruct signal by adding up the weighted sinc functions. The
    # weighted sinc functions are computed on the fly here to save
    # memory:
    t = arange(0, dur, dt)
    u_rec = zeros(len(t), float)
    c = dot(G_inv, q)
    for i in xrange(Nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def asdm_decode_ins(s, dur, dt, bw, b, sgn=-1):    
    """
    Decode a signal encoded with an ASDM encoder using a
    threshold-insensitive recovery algorithm.

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
    sgn : {-1, 1}
        Sign of first spike.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.

    """
    
    Ns = len(s)
    if Ns < 2:
        raise ValueError('s must contain at least 2 elements') 

    # Cast s to an ndarray to permit ndarray operations:
    s = asarray(s)

    # Compute the spike times:
    ts = cumsum(s)

    # Compute the midpoints between spike times:
    tsh = (ts[0:-1]+ts[1:])/2
    Nsh = len(tsh)
    
    t = arange(0, dur, dt)
    
    bwpi = bw/pi
    
    # Compute G matrix:
    G = empty((Nsh, Nsh), float)
    for j in xrange(Nsh):

        # Compute the values for all of the sinc functions so that
        # they do not need to each be recomputed when determining the
        # integrals between spike times:
        temp = sici(bw*(ts-tsh[j]))[0]/pi
        for i in xrange(Nsh):
            G[i, j] = temp[i+1]-temp[i]
    
    # Apply compensation principle:
    B = diag(ones(Nsh-1), -1)+eye(Nsh)
    if sgn == -1:
        Bq = array([(-1)**i for i in xrange(Nsh)])*b*(s[1:]-s[:-1])
    else:
        Bq = array([(-1)**i for i in xrange(1, Nsh+1)])*b*(s[1:]-s[:-1])
        
    # Reconstruct signal by adding up the weighted sinc functions; the
    # first row of B is removed to eliminate boundary issues. The
    # weighted sinc functions are computed on the fly to save memory:
    u_rec = zeros(len(t), float)
    c = dot(pinv(dot(B[1:, :], G), __pinv_rcond__), Bq[1:, newaxis])
    for i in xrange(Nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def asdm_decode_fast(s, dur, dt, bw, M, b, d, k=1.0, sgn=-1):
    """
    Decode a signal encoded by an ASDM encoder using a fast recovery
    algorithm.

    Parameters
    ----------
    s : numpy array of floats
        Encoded signal. The values represent the time between spikes (in s).
    dur : float
        Duration of signal (in s).
    dt : float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw : float
        Signal bandwidth (in rad/s).
    M : int
        Number of bins used by the fast algorithm.
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    k : float
        Encoder integrator constant.
    sgn : {-1, 1}
        Sign of first spike.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.

    """
    
    Ns = len(s)
    if Ns < 2:
        raise ValueError('s must contain at least 2 elements')

    # Cast s to an ndarray to permit ndarray operations:
    s = asarray(s)

    # Compute the spike times:
    ts = cumsum(s)

    # Compute the midpoints between spike times:
    tsh = (ts[0:-1]+ts[1:])/2
    Nsh = len(tsh)

    # Convert M in the event that an integer was specified:
    M = float(M)
    jbwM = 1j*bw/M

    # Compute quanta:
    if sgn == -1:
        q = array([(-1)**i for i in xrange(1, Nsh+1)])*(2*k*d-b*s[1:])
    else:
        q = array([(-1)**i for i in xrange(0, Nsh)])*(2*k*d-b*s[1:])
        
    # Compute approximation coefficients:
    a = bw/(pi*(2*M+1))
    m = arange(-M, M+1)
    P_inv = -triu(ones((Nsh, Nsh)))
    S = exp(-jbwM*dot(m[:, newaxis], ts[:-1][newaxis]))
    D = diag(s[1:])
    SD = dot(S, D)
    T = mdot(a, SD, conjugate(S.T))
    dd = mdot(a, pinv(T, __pinv_rcond__), SD, P_inv, q[:, newaxis])

    # Reconstruct signal:
    t = arange(0, dur, dt)
    return ravel(real(jbwM*dot(m*dd.T, exp(jbwM*m[:, newaxis]*t))))

def asdm_decode_pop(s_list, dur, dt, bw, b_list, d_list, k_list, sgn_list=[]):
    """
    Decode a signal encoded by an ensemble of ASDM encoders.

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
    k_list : list of floats
        List of encoder integration constants.
    sgn_list : list of integers {-1, 1}
        List of signs of first spikes in trains.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.

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
    
    # Compute the spike times:
    ts_list = map(cumsum, s_list)

    # Compute the midpoints between spike times:
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
    """
    Decode a signal encoded by an ensemble of ASDM encoders using a
    threshold-insensitive recovery algorithm.

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

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.

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
    
    # Compute the spike times:
    ts_list = map(cumsum, s_list)

    # Compute the midpoints between spike times:
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

