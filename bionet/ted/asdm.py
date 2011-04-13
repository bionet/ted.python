#!/usr/bin/env python

"""
Time encoding and decoding algorithms that make use of the
asynchronous sigma-delta modulator.
 
- asdm_decode         - ASDM time decoding machine.
- asdm_decode_fast    - Fast ASDM time decoding machine.
- asdm_decode_ins     - Threshold-insensitive ASDM time decoding machine.
- asdm_decode_pop     - MISO ASDM time decoding machine.
- asdm_decode_pop_ins - Threshold-insensitive MISO ASDM time decoding machine.
- asdm_encode         - ASDM time encoding machine.
- asdm_recoverable    - ASDM time encoding parameter check.

"""

__all__ = ['asdm_recoverable', 'asdm_encode', 'asdm_decode',
           'asdm_decode_ins', 'asdm_decode_fast',
	   'asdm_decode_pop', 'asdm_decode_pop_ins']

import numpy as np
import scipy.signal

# The sici() function in scipy.special is used to construct the matrix
# G in certain decoding algorithms because it can compute the sine
# integral relatively quickly:
import scipy.special

import bionet.utils.numpy_extras as ne
from bionet.ted.vtdm import asdm_decode_vander, \
     asdm_decode_vander_ins

__all__ += ['asdm_decode_vander', 'asdm_decode_vander_ins']

# Pseudoinverse singular value cutoff:
__pinv_rcond__ = 1e-8

def asdm_recoverable_strict(u, bw, b, d, k):
    """
    ASDM time encoding parameter check.
    
    Determine whether a signal encoded with an Asynchronous
    Sigma-Delta Modulator using the specified parameters can be
    perfectly recovered.
    
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
    
    c = np.max(np.abs(u))
    if c >= b:
        raise ValueError('bias too low')
    elif 2*k*d/(b-c)*bw/np.pi >= 1.0:
        raise ValueError('reconstruction condition not satisfied;'+
                         'try raising b or reducing k or d')
    else:
        return True

def asdm_recoverable(u, bw, b, d, k):
    """
    ASDM time encoding parameter check.

    Determine whether a signal encoded with an Asynchronous
    Sigma-Delta Modulator using the specified parameters can be
    perfectly recovered.

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
        
    Notes
    -----
    The bound assumed by this check is not as strict as that described in
    most of Prof. Lazar's papers.

    """
    
    c = np.max(np.abs(u))
    if c >= b:
        raise ValueError('bias too low')
    elif (2*k*d/b)*bw/np.pi >= 1.0:
        raise ValueError('reconstruction condition not satisfied;'+
                         'try raising b or reducing k or d')
    else:
        return True

def asdm_encode(u, dt, b, d, k=1.0, dte=0.0, y=0.0, interval=0.0,
                sgn=1, quad_method='trapz', full_output=False):
    """
    ASDM time encoding machine.

    Encode a finite length signal using an Asynchronous Sigma-Delta
    Modulator.
    
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
            return np.array((), np.float), dt, b, d, k, dte, y, interval, sgn, \
               quad_method, full_output
        else:
            return np.array((), np.float)

    # Check whether the encoding resolution is finer than that of the
    # original sampled signal:
    if dte > dt:
        raise ValueError('encoding time resolution must not exceeed original signal resolution')
    if dte < 0:
        raise ValueError('encoding time resolution must be nonnegative')
    if dte != 0 and dte != dt:

        # Resample signal and adjust signal length accordingly:
        M = int(dt/dte)
        u = scipy.signal.resample(u, len(u)*M)
        Nu *= M
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
        if np.abs(y) >= d:
            s.append(interval)
            interval = 0.0
            y = d*sgn
            sgn = -sgn

    if full_output:
        return np.array(s), dt, b, d, k, dte, y, interval, sgn, \
               quad_method, full_output
    else:
        return np.array(s)

def asdm_decode(s, dur, dt, bw, b, d, k=1.0, sgn=-1):    
    """
    ASDM time decoding machine.
    
    Decode a signal encoded with an Asynchronous Sigma-Delta Modulator.

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
    s = np.asarray(s)

    # Compute the spike times:
    ts = np.cumsum(s)

    # Compute the midpoints between spike times:
    tsh = (ts[0:-1]+ts[1:])/2
    Nsh = len(tsh)
    
    bwpi = bw/np.pi
    
    # Compute G matrix:
    G = np.empty((Nsh, Nsh), np.float)
    for j in xrange(Nsh):

        # Compute the values for all of the sincs so that they do not
        # need to each be recomputed when determining the integrals
        # between spike times:
        temp = scipy.special.sici(bw*(ts-tsh[j]))[0]/np.pi
        G[:, j] = temp[1:]-temp[:-1]
    G_inv = np.linalg.pinv(G, __pinv_rcond__)

    # Compute quanta:
    if sgn == -1:
        q = (-1)**np.arange(1, Nsh+1)*(2*k*d-b*s[1:])
    else:
        q = (-1)**np.arange(0, Nsh)*(2*k*d-b*s[1:])
        
    # Reconstruct signal by adding up the weighted sinc functions. The
    # weighted sinc functions are computed on the fly here to save
    # memory:
    t = np.arange(0, dur, dt)
    u_rec = np.zeros(len(t), np.float)
    c = np.dot(G_inv, q)
    for i in xrange(Nsh):
        u_rec += np.sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def asdm_decode_ins(s, dur, dt, bw, b, sgn=-1):    
    """
    Threshold-insensitive ASDM time decoding machine.
    
    Decode a signal encoded with an Asynchronous Sigma-Delta
    Modulator using a threshold-insensitive recovery algorithm.

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
    s = np.asarray(s)

    # Compute the spike times:
    ts = np.cumsum(s)

    # Compute the midpoints between spike times:
    tsh = (ts[0:-1]+ts[1:])/2
    Nsh = len(tsh)
    
    t = np.arange(0, dur, dt)
    
    bwpi = bw/np.pi
    
    # Compute G matrix:
    G = np.empty((Nsh, Nsh), np.float)
    for j in xrange(Nsh):

        # Compute the values for all of the sinc functions so that
        # they do not need to each be recomputed when determining the
        # integrals between spike times:
        temp = scipy.special.sici(bw*(ts-tsh[j]))[0]/np.pi
        G[:, j] = temp[1:]-temp[:-1]
    
    # Apply compensation principle:
    B = np.diag(np.ones(Nsh-1), -1)+np.eye(Nsh)
    if sgn == -1:
        Bq = (-1)**np.arange(Nsh)*b*(s[1:]-s[:-1])
    else:
        Bq = (-1)**np.arange(1, Nsh+1)*b*(s[1:]-s[:-1])
        
    # Reconstruct signal by adding up the weighted sinc functions; the
    # first row of B is removed to eliminate boundary issues. The
    # weighted sinc functions are computed on the fly to save memory:
    u_rec = np.zeros(len(t), np.float)
    c = np.dot(np.linalg.pinv(np.dot(B[1:, :], G), __pinv_rcond__), Bq[1:, np.newaxis])
    for i in xrange(Nsh):
        u_rec += np.sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def asdm_decode_fast(s, dur, dt, bw, M, b, d, k=1.0, sgn=-1):
    """
    Fast ASDM time decoding machine.
    
    Decode a signal encoded by an Asynchronous Sigma-Delta Modulator
    using a fast recovery algorithm.

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
    s = np.asarray(s)

    # Compute the spike times:
    ts = np.cumsum(s)

    # Compute the midpoints between spike times:
    tsh = (ts[0:-1]+ts[1:])/2
    Nsh = len(tsh)

    # Convert M in the event that an integer was specified:
    M = np.float(M)
    jbwM = 1j*bw/M

    # Compute quanta:
    if sgn == -1:
        q = (-1)**np.arange(1, Nsh+1)*(2*k*d-b*s[1:])
    else:
        q = (-1)**np.arange(0, Nsh)*(2*k*d-b*s[1:])
        
    # Compute approximation coefficients:
    a = bw/(np.pi*(2*M+1))
    m = np.arange(-M, M+1)
    P_inv = -np.triu(np.ones((Nsh, Nsh)))
    S = np.exp(-jbwM*np.dot(m[:, np.newaxis], ts[:-1][np.newaxis]))
    D = np.diag(s[1:])
    SD = np.dot(S, D)
    T = ne.mdot(a, SD, np.conj(S.T))
    dd = ne.mdot(a, np.linalg.pinv(T, __pinv_rcond__), SD, P_inv, q[:, np.newaxis])

    # Reconstruct signal:
    t = np.arange(0, dur, dt)
    return np.ravel(np.real(jbwM*np.dot(m*dd.T, np.exp(jbwM*m[:, np.newaxis]*t))))

def asdm_decode_pop(s_list, dur, dt, bw, b_list, d_list, k_list, sgn_list=[]):
    """
    Multi-input single-output ASDM time decoding machine.
    
    Decode a signal encoded by an ensemble of Asynchronous Sigma-Delta
    Modulators.

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

    bwpi = bw/np.pi
    
    # Compute the spike times:
    ts_list = map(np.cumsum, s_list)

    # Compute the midpoints between spike times:
    tsh_list = map(lambda ts:(ts[0:-1]+ts[1:])/2, ts_list)

    # Compute number of spikes in each spike list:
    Ns_list = map(len, ts_list)
    Nsh_list = map(len, tsh_list)
        
    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    Nsh_cumsum = np.cumsum([0.0]+Nsh_list)
    Nsh_sum = Nsh_cumsum[-1]
    G = np.empty((Nsh_sum, Nsh_sum), np.float)
    q = np.empty((Nsh_sum, 1), np.float)
    for l in xrange(M):
        for m in xrange(M):
            G_block = np.empty((Nsh_list[l], Nsh_list[m]), np.float)

            # Compute the values for all of the sincs so that they
            # do not need to each be recomputed when determining
            # the integrals between spike times:
            for k in xrange(Nsh_list[m]):
                temp = scipy.special.sici(bw*(ts_list[l]-tsh_list[m][k]))[0]/np.pi
                G_block[:, k] = temp[1:]-temp[:-1]

            G[Nsh_cumsum[l]:Nsh_cumsum[l+1],
              Nsh_cumsum[m]:Nsh_cumsum[m+1]] = G_block

        # Compute the quanta:
        if sgn_list[l] == -1:
            q[Nsh_cumsum[l]:Nsh_cumsum[l+1], 0] = \
                (-1)**np.arange(1, Nsh_list[l]+1)* \
                (2*k_list[l]*d_list[l]-b_list[l]*s_list[l][1:])
        else:
            q[Nsh_cumsum[l]:Nsh_cumsum[l+1], 0] = \
                (-1)**np.arange(0, Nsh_list[l])* \
                (2*k_list[l]*d_list[l]-b_list[l]*s_list[l][1:])
            
    # Compute the reconstruction coefficients:
    c = np.dot(np.linalg.pinv(G), q)

    # Reconstruct the signal using the coefficients:
    t = np.arange(0, dur, dt)
    u_rec = np.zeros(len(t), np.float)
    for m in xrange(M):
        for k in xrange(Nsh_list[m]):
            u_rec += np.sinc(bwpi*(t-tsh_list[m][k]))*bwpi*c[Nsh_cumsum[m]+k, 0]
    return u_rec

def asdm_decode_pop_ins(s_list, dur, dt, bw, b_list, sgn_list=[]):
    """
    Threshold-insensitive multi-input single-output time decoding
    machine.
    
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

    bwpi = bw/np.pi
    
    # Compute the spike times:
    ts_list = map(np.cumsum, s_list)

    # Compute the midpoints between spike times:
    tsh_list = map(lambda ts:(ts[0:-1]+ts[1:])/2, ts_list)

    # Compute number of spikes in each spike list:
    Nsh_list = map(lambda x: len(x)-1, tsh_list)
    
    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    Nsh_cumsum = np.cumsum([0.0]+Nsh_list)
    Nsh_sum = Nsh_cumsum[-1]
    G = np.empty((Nsh_sum, Nsh_sum), np.float)
    Bq = np.empty((Nsh_sum, 1), np.float)
    for l in xrange(M):
        for m in xrange(M):
            G_block = np.empty((Nsh_list[l], Nsh_list[m]), np.float)

            # Compute the values for all of the sincs so that they
            # do not need to each be recomputed when determining
            # the integrals between spike times:
            for k in xrange(Nsh_list[m]):
                temp = scipy.special.sici(bw*(ts_list[l]-tsh_list[m][k]))[0]/np.pi
                G_block[:, k] = temp[2:]-temp[:-2]

            G[Nsh_cumsum[l]:Nsh_cumsum[l+1],
              Nsh_cumsum[m]:Nsh_cumsum[m+1]] = G_block

        # Compute the quanta:
        if sgn_list[l] == -1:
            Bq[Nsh_cumsum[l]:Nsh_cumsum[l+1], 0] = \
                (-1)**np.arange(1, Nsh_list[l]+1)* \
                b_list[l]*(s_list[l][2:]-s_list[l][1:-1])
        else:
            Bq[Nsh_cumsum[l]:Nsh_cumsum[l+1], 0] = \
                (-1)**np.arange(0, Nsh_list[l])* \
                b_list[l]*(s_list[l][2:]-s_list[l][1:-1])

    # Compute the reconstruction coefficients:
    c = np.dot(np.linalg.pinv(G), Bq)

    # Reconstruct the signal using the coefficients:
    t = np.arange(0, dur, dt)
    u_rec = np.zeros(len(t), np.float)
    for m in xrange(M):
        for k in xrange(Nsh_list[m]):
            u_rec += np.sinc(bwpi*(t-tsh_list[m][k]))*bwpi*c[Nsh_cumsum[m]+k, 0]
    return u_rec

