#!/usr/bin/env python

"""
Time encoding and decoding algorithms that make use of the
integrate-and-fire neuron model.

- iaf_decode            - Decode a signal encoded by an IAF encoder.
- iaf_decode_fast       - Fast IAF decoding algorithm.
- iaf_decode_pop        - Decode a signal encoded by several IAF encoders.
- iaf decode_coupled    - Decode a signal encoded by coupled IAF encoders.
- iaf_decode_delay      - Decode a signal encoded by delayed IAF encoders.
- iaf_decode_spline     - Decode a IAF-encoded signal using spline interpolation.
- iaf_decode_spline_pop - iaf_decode_spline for a population of IAF encoders.
- iaf_encode            - Encode a signal using an IAF encoder.
- iaf_encode_delay      - Encode a signal using delayed IAF encoders.
- iaf_encode_coupled    - Encode a signal using coupled IAF encoders.
- iaf_recoverable       - Check encoder parameters for decoding feasibility.

"""

__all__ = ['iaf_recoverable', 'iaf_encode', 'iaf_decode',
           'iaf_decode_fast', 'iaf_decode_pop',
           'iaf_decode_spline', 'iaf_decode_spline_pop',
           'iaf_encode_coupled', 'iaf_decode_coupled',
           'iaf_encode_delay', 'iaf_decode_delay']

# Import max() as amax() because the builtin max() function is needed
# by iaf_decode_spline_pop():
from numpy import abs, all, amax, arange, array, asarray, conjugate, cumsum, \
     diag, diff, dot, empty, exp, eye, float, hstack, imag, inf, \
     isinf, isreal, log, newaxis, nonzero, ones, pi, ravel, \
     real, shape, sinc, sum, triu, where, zeros
from numpy.linalg import pinv
from scipy.integrate import quad
from scipy.signal import resample

# The sici() and ei() functions are used to construct the decoding
# matrix G because they can respectively compute the sine and
# exponential integrals relatively quickly:
from scipy.special import sici
from bionet.utils.numpy_extras import mdot
from bionet.utils.scipy_extras import ei

from bionet.ted.vtdm import iaf_decode_vander

__all__ += ['iaf_decode_vander']

# Pseudoinverse singular value cutoff:
__pinv_rcond__ = 1e-8

def iaf_recoverable(u, bw, b, d, R, C):
    """
    Determine whether a signal can be perfectly recovered using an IAF
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
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.

    Returns
    -------
    rec : bool
        True if the specified signal is recoverable.
        
    Raises
    ------
    ValueError
        When the signal cannot be perfectly recovered.

    """

    c = amax(abs(u))
    if c >= b:
        raise ValueError('bias too low')
    r = R*C*log(1-d/(d-(b-c)*R))*bw/pi
    e = d/((b-c)*R)
    if not isreal(r):
        raise ValueError('reconstruction condition not satisfied')
    elif r >= (1-e)/(1+e):
        raise ValueError('reconstruction condition not satisfied;'+
                         'try raising b or reducing d')
    else:
        return True

def iaf_encode(u, dt, b, d, R=inf, C=1.0, dte=0, y=0.0, interval=0.0,
               quad_method='trapz', full_output=False):
    """
    Encode a signal with an IAF neuron.

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
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.
    dte : float
        Sampling resolution assumed by the encoder (s).
        This may not exceed `dt`.
    y : float
        Initial value of integrator.
    interval : float
        Time since last spike (in s).
    quad_method : {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal) when the
        neuron is not leaky; exponential Euler integration is used
        when the neuron is leaky.
    full_output : bool
        If set, the function returns the encoded data block followed
        by the given parameters (with updated values for `y` and `interval`).
        This is useful when the function is called repeatedly to
        encode a long signal.

    Returns
    -------
    s : ndarray of floats
        If `full_output` == False, returns the signal encoded as an
        array of time intervals between spikes.
    s, dt, b, d, R, C, dte, y, interval, quad_method, full_output : tuple
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
            return array((),float), dt, b, d, R, C, dte, y, interval, \
                   quad_method, full_output
        else:
            return array((),float)
    
    # Check whether the encoding resolution is finer than that of the
    # original sampled signal:
    if dte > dt:
        raise ValueError('encoding time resolution must not exceeed original signal resolution')
    if dte < 0:
        raise ValueError('encoding time resolution must be nonnegative')
    if dte != 0:        
        u = resample(u, len(u)*int(dt/dte))
        dt = dte

    # Use a list rather than an array to save the spike intervals
    # because the number of spikes is not fixed:
    s = []

    # Choose integration method:
    if isinf(R):        
        if quad_method == 'rect':
            compute_y = lambda y, i: y + dt*(b+u[i])/C
            last = Nu
        elif quad_method == 'trapz':
            compute_y = lambda y, i: y + dt*(b+(u[i]+u[i+1])/2.0)/C
            last = Nu-1
        else:
            raise ValueError('unrecognized quadrature method')
    else:

        # When the neuron is leaky, use the exponential Euler method to perform
        # the encoding:
        RC = R*C
        compute_y = lambda y, i: y*exp(-dt/RC)+R*(1-exp(-dt/RC))*(b+u[i])
        last = Nu
        
    # The interval between spikes is saved between iterations rather than the
    # absolute time so as to avoid overflow problems for very long signals:
    for i in xrange(last):
        y = compute_y(y, i)
        interval += dt
        if y >= d:
            s.append(interval)
            interval = 0.0
            y = 0.0

    if full_output:
        return array(s), dt, b, d, R, C, dte, y, interval, \
               quad_method, full_output
    else:
        return array(s)

def iaf_decode(s, dur, dt, bw, b, d, R=inf, C=1.0):
    """
    Decode a signal encoded with an IAF neuron.

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
    RC = R*C

    # Compute G matrix and quanta:
    G = empty((Nsh,Nsh),float)
    if isinf(R):
        for j in xrange(Nsh):
            temp = sici(bw*(ts-tsh[j]))[0]/pi
            for i in xrange(Nsh):
                G[i,j] = temp[i+1]-temp[i]
        q = C*d-b*s[1:]
    else:
        for i in xrange(Nsh):            
            for j in xrange(Nsh):

                # The code below is functionally equivalent to (but
                # considerably faster than) the integration below:
                #
                # f = lambda t:sinc(bwpi*(t-tsh[j]))*bwpi*exp((ts[i+1]-t)/-RC)
                # G[i,j] = quad(f, ts[i], ts[i+1])[0]
                if ts[i] < tsh[j] and tsh[j] < ts[i+1]:
                    G[i,j] = (-1j/4)*exp((tsh[j]-ts[i+1])/RC)* \
                             (2*ei((1-1j*RC*bw)*(ts[i]-tsh[j])/RC)-
                              2*ei((1-1j*RC*bw)*(ts[i+1]-tsh[j])/RC)-
                              2*ei((1+1j*RC*bw)*(ts[i]-tsh[j])/RC)+
                              2*ei((1+1j*RC*bw)*(ts[i+1]-tsh[j])/RC)+
                              log(-1-1j*RC*bw)+log(1-1j*RC*bw)-
                              log(-1+1j*RC*bw)-log(1+1j*RC*bw)+
                              log(-1j/(-1j+RC*bw))-log(1j/(-1j+RC*bw))+
                              log(-1j/(1j+RC*bw))-log(1j/(1j+RC*bw)))/pi
                else:
                    G[i,j] = (-1j/2)*exp((tsh[j]-ts[i+1])/RC)* \
                             (ei((1-1j*RC*bw)*(ts[i]-tsh[j])/RC)-
                              ei((1-1j*RC*bw)*(ts[i+1]-tsh[j])/RC)-
                              ei((1+1j*RC*bw)*(ts[i]-tsh[j])/RC)+
                              ei((1+1j*RC*bw)*(ts[i+1]-tsh[j])/RC))/pi
                    
        q = C*(d+b*R*(exp(-s[1:]/RC)-1))

    # Compute the reconstruction coefficients:
    c = dot(pinv(G, __pinv_rcond__), q)
    
    # Reconstruct signal by adding up the weighted sinc functions.
    u_rec = zeros(len(t), float)
    for i in xrange(Nsh):
        u_rec += sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec

def iaf_decode_fast(s, dur, dt, bw, M, b, d, R=inf, C=1.0):
    """
    Decode a signal encoded with an IAF neuron using a fast recovery
    algorithm.

    Parameters
    ----------
    s : array_like of floats
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
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.

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

    # Convert M to a float in the event that an integer was specified:
    M = float(M)

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
    P_inv = -triu(ones((Nsh,Nsh)))
    S = exp(-jbwM*dot(m[:, newaxis], ts[:-1][newaxis]))
    D = diag(s[1:])
    SD = dot(S,D)
    T = mdot(a, SD,conjugate(S.T))
    dd = mdot(a, pinv(T, __pinv_rcond__), SD, P_inv, q[:,newaxis])

    # Reconstruct signal:
    t = arange(0, dur, dt)
    return ravel(real(jbwM*dot(m*dd.T, exp(jbwM*m[:, newaxis]*t))))

def iaf_decode_pop(s_list, dur, dt, bw, b_list, d_list, R_list, C_list):
    """
    Decode a signal encoded with an ensemble of IAF neurons.

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
    if all(isinf(R_list)):
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
            q[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]), 0] = \
                       C_list[l]*d_list[l]-b_list[l]*s_list[l][1:]
    else:
        for l in xrange(M):
            for m in xrange(M):
                G_block = empty((Nsh_list[l], Nsh_list[m]), float)

                for n in xrange(Nsh_list[l]):
                    for k in xrange(Nsh_list[m]):

                        # The code below is functionally equivalent to
                        # (but considerably faster than) the
                        # integration below:
                        #
                        # f = lambda t:sinc(bwpi*(t-tsh_list[m][k]))* \
                        #     bwpi*exp((ts_list[l][n+1]-t)/-(R_list[l]*C_list[l]))
                        # G_block[n, k] = quad(f, ts_list[l][n], ts_list[l][n+1])[0]
                        RC = R_list[l]*C_list[l]
                        tsh = tsh_list[m]
                        ts = ts_list[l]
                        if ts[n] < tsh[k] and tsh[k] < ts[n+1]:
                            G_block[n, k] = (-1j/4)*exp((tsh[k]-ts[n+1])/(RC))* \
                                            (2*ei((1-1j*RC*bw)*(ts[n]-tsh[k])/RC)-
                                             2*ei((1-1j*RC*bw)*(ts[n+1]-tsh[k])/RC)-
                                             2*ei((1+1j*RC*bw)*(ts[n]-tsh[k])/RC)+
                                             2*ei((1+1j*RC*bw)*(ts[n+1]-tsh[k])/RC)+
                                             log(-1-1j*RC*bw)+log(1-1j*RC*bw)-
                                             log(-1+1j*RC*bw)-log(1+1j*RC*bw)+
                                             log(-1j/(-1j+RC*bw))-log(1j/(-1j+RC*bw))+
                                             log(-1j/(1j+RC*bw))-log(1j/(1j+RC*bw)))/pi
                        else:
                            G_block[n, k] = (-1j/2)*exp((tsh[k]-ts[n+1])/RC)* \
                                            (ei((1-1j*RC*bw)*(ts[n]-tsh[k])/RC)-
                                             ei((1-1j*RC*bw)*(ts[n+1]-tsh[k])/RC)-
                                             ei((1+1j*RC*bw)*(ts[n]-tsh[k])/RC)+
                                             ei((1+1j*RC*bw)*(ts[n+1]-tsh[k])/RC))/pi   

                G[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]),
                  sum(Nsh_list[:m]):sum(Nsh_list[:m+1])] = G_block

            # Compute the quanta:
            q[sum(Nsh_list[:l]):sum(Nsh_list[:l+1]), 0] = \
                       C_list[l]*(d_list[l]+b_list[l]*R_list[l]* \
                                  (exp(-s_list[l][1:]/(R_list[l]*C_list[l]))-1))
    
    # Compute the reconstruction coefficients:
    c = dot(pinv(G, __pinv_rcond__), q)

    # Reconstruct the signal using the coefficients:
    t = arange(0, dur, dt)
    u_rec = zeros(len(t), float)
    for m in xrange(M):
        for k in xrange(Nsh_list[m]):
            u_rec += sinc(bwpi*(t-tsh_list[m][k]))*bwpi*c[sum(Nsh_list[:m])+k, 0]
    return u_rec

def iaf_decode_spline(s, dur, dt, b, d, R=inf, C=1.0):
    """
    Decode a signal encoded with an IAF neuron using spline
    interpolation.

    Parameters
    ----------
    s : array_like of floats
        Encoded signal. The values represent the time between spikes (in s).
    dur : float
        Duration of signal (in s).
    dt : float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.

    """
    
    ns = len(s)
    if ns < 2:
        raise ValueError('s must contain at least 2 elements')

    # Cast s to an ndarray to permit ndarray operations:
    s = asarray(s)

    # Compute the spike times:
    ts = cumsum(s)
    n = ns-1

    RC = R*C
    
    # Define the spline polynomials:
    f = lambda x: x**3-3*x**2+6*x-6
    g = lambda x: x**3+6*x

    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    Gpr = zeros((n+2, n+2), float)
    qz = zeros(n+2, float)
    if isinf(R):

        # Compute p and r:
        Gpr[n, :n] = Gpr[:n, n] = s[1:]
        Gpr[n+1, :n] = Gpr[:n, n+1] = (ts[1:]**2-ts[:-1]**2)/2

        # Compute the quanta:
        qz[:n] = C*d-b*s[1:]

        # Compute the matrix G:
        for k in xrange(n):
            for l in xrange(n):
                if k < l:
                    Gpr[k, l] = 0.05*((ts[k+1]-ts[l+1])**5+\
                                    (ts[k]-ts[l])**5-\
                                    (ts[k]-ts[l+1])**5-\
                                    (ts[k+1]-ts[l])**5)                                   
                elif k == l:
                    Gpr[k, l] = 0.1*(ts[k+1]-ts[k])**5    
                else:
                    Gpr[k, l] = 0.05*((ts[l+1]-ts[k+1])**5+\
                                    (ts[l]-ts[k])**5-\
                                    (ts[l]-ts[k+1])**5-\
                                    (ts[l+1]-ts[k])**5)                                   
                    
    else:

        # Compute p and r:
        Gpr[n, :n] = Gpr[:n, n] = RC*(1-exp(-s[1:]/RC))
        Gpr[n+1, :n] = Gpr[:n, n+1] = RC**2*((ts[1:]/RC-1)-(ts[:-1]/RC-1)*exp(-s[1:]/RC))

        # Compute the quanta:
        qz[:n] = C*(d-b*R*(1-exp(-s[1:]/RC)))

        # Compute the matrix G:
        for k in xrange(n):
            for l in xrange(n):
                if k < l:
                    Gpr[k, l] = RC**5*(g((ts[l+1]-ts[k+1])/RC)-\
                                     g((ts[l+1]-ts[k])/RC)*exp(-(ts[k+1]-ts[k])/RC)-\
                                     g((ts[l]-ts[k+1])/RC)*exp(-(ts[l+1]-ts[l])/RC)+\
                                     g((ts[l]-ts[k])/RC)*exp(-(ts[k+1]-ts[k])/RC-(ts[l+1]-ts[l])/RC))
                elif k == l:
                    Gpr[k, l] = RC**5*(6*(1-exp(-2*(ts[k+1]-ts[k])/RC))-\
                                     2*g((ts[k+1]-ts[k])/RC)*exp(-(ts[k+1]-ts[k])/RC))
                else:
                    Gpr[k, l] = RC**5*(g((ts[k+1]-ts[l+1])/RC)-\
                                     g((ts[k+1]-ts[l])/RC)*exp(-(ts[l+1]-ts[l])/RC)-\
                                     g((ts[k]-ts[l+1])/RC)*exp(-(ts[k+1]-ts[k])/RC)+\
                                     g((ts[k]-ts[l])/RC)*exp(-(ts[l+1]-ts[l])/RC-(ts[k+1]-ts[k])/RC))
                                     
    # Compute the reconstruction coefficients:
    ## NOTE: setting the svd cutoff higher than 10**-15 appears to
    ## introduce considerable recovery error:
    cd = dot(pinv(Gpr), qz)

    # Reconstruct the signal using the coefficients:
    t = arange(0, dur, dt)
    u_rec = cd[n] + cd[n+1]*t
    if isinf(R):
        psi = lambda t, k: \
              0.25*where(t <= ts[k],
                         ((t-ts[k+1])**4-(t-ts[k])**4),
                         where(t <= ts[k+1],
                               ((t-ts[k+1])**4+(t-ts[k])**4),
                               ((t-ts[k])**4-(t-ts[k+1])**4)))
    else:
        psi = lambda t, k: \
              (RC)**4*where(t <= ts[k],
                            f((ts[k+1]-t)/RC)-f((ts[k]-t)/RC)*exp(-(ts[k+1]-ts[k])/RC),
                            where(t <= ts[k+1],
                                  12*exp(-(ts[k+1]-t)/RC)+f((ts[k+1]-t)/RC)+
                                  f((ts[k]-t)/RC)*exp(-(ts[k+1]-ts[k])/RC),
                                  f((ts[k]-t)/RC)*exp(-(ts[k+1]-ts[k])/RC)-f((ts[k+1]-t)/RC)))
    for k in xrange(n):
        u_rec += cd[k]*psi(t, k)
                                     
    return u_rec

def iaf_decode_spline_pop(s_list, dur, dt, b_list, d_list, R_list,
                          C_list):
    """
    Decode a signal encoded with an ensemble of IAF neurons using
    spline interpolation.

    Parameters
    ----------
    s_list: list of ndarrays of floats
        Signal encoded by an ensemble of encoders. The values represent the
        time between spikes (in s). The number of arrays in the list
        corresponds to the number of encoders in the ensemble.
    dur: float
        Duration of signal (in s).
    dt: float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    b_list: list of floats
        List of encoder biases.
    d_list: list of floats
        List of encoder thresholds.
    R_list: list of floats
        List of encoder neuron resistances.
    C_list: list of floats.    

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

    # Compute the spike times:
    ts_list = map(cumsum, s_list)
    n_list = map(lambda ts: len(ts)-1, ts_list)

    # Define the spline polynomial:
    f = lambda x: x**3-3*x**2+6*x-6

    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    n_sum = sum(n_list)
    Gpr = zeros((n_sum+2, n_sum+2), float)
    qz = zeros(n_sum+2, float)    
    if all(isinf(R_list)):
        for i in xrange(M):

            # Compute p and r:
            s = s_list[i]
            ts = ts_list[i]
            Gpr[n_sum, sum(n_list[:i]):sum(n_list[:i+1])] = \
                       Gpr[sum(n_list[:i]):sum(n_list[:i+1]), n_sum] = s[1:]
            Gpr[n_sum+1, sum(n_list[:i]):sum(n_list[:i+1])] = \
                         Gpr[sum(n_list[:i]):sum(n_list[:i+1]), n_sum+1] = \
                         (ts[1:]**2-ts[:-1]**2)/2

            # Compute the quanta:
            qz[sum(n_list[:i]):sum(n_list[:i+1])] = \
                C_list[i]*d_list[i]-b_list[i]*s[1:]
            
            # Compute the G matrix:
            for j in xrange(M):
                Gpr_block = zeros((n_list[i], n_list[j]), float)
                for k in xrange(n_list[i]):
                    for l in xrange(n_list[j]):
                        a1 = ts_list[i][k]
                        b1 = min(ts_list[j][l], ts_list[i][k+1])
                        a2 = max(ts_list[j][l], ts_list[i][k])
                        b2 = min(ts_list[j][l+1], ts_list[i][k+1])
                        a3 = max(ts_list[j][l+1], ts_list[i][k])
                        b3 = ts_list[i][k+1]
                        
                        # The analytic expression for Gpr_block[k, l] is equivalent
                        # to the integration described in the comment below:
                        # f1 = lambda t: \
                        #      0.25*(((t-ts_list[j][l+1])**4-(t-ts_list[j][l])**4))
                        # f2 = lambda t: \
                        #      0.25*(((t-ts_list[j][l+1])**4+(t-ts_list[j][l])**4))
                        # f3 = lambda t: \
                        #      0.25*(((t-ts_list[j][l])**4-(t-ts_list[j][l+1])**4))
                        # if (ts_list[i][k]<ts_list[j][l]):
                        #     Gpr_block[k, l] += quad(f1, a1, b1)[0]
                        # if (ts_list[j][l]<ts_list[i][k+1] and
                        #     ts_list[j][l+1]>ts_list[i][k]):
                        #     Gpr_block[k, l] += quad(f2, a2, b2)[0]
                        # if (ts_list[j][l+1]<ts_list[i][k+1]):
                        #     Gpr_block[k, l] += quad(f3, a3, b3)[0]
                        if (ts_list[i][k]<ts_list[j][l]):
                            Gpr_block[k, l] += \
                                         0.05*(((b1-ts_list[j][l+1])**5-(b1-ts_list[j][l])**5)\
                                               -((a1-ts_list[j][l+1])**5-(a1-ts_list[j][l])**5))
                        if (ts_list[j][l]<ts_list[i][k+1] and ts_list[j][l+1]>ts_list[i][k]):
                            Gpr_block[k, l] += \
                                         0.05*(((b2-ts_list[j][l+1])**5+(b2-ts_list[j][l])**5)\
                                               -((a2-ts_list[j][l+1])**5+(a2-ts_list[j][l])**5))
                        if (ts_list[j][l+1]<ts_list[i][k+1]):
                            Gpr_block[k, l] += \
                                         0.05*(((b3-ts_list[j][l])**5-(b3-ts_list[j][l+1])**5)\
                                               -((a3-ts_list[j][l])**5-(a3-ts_list[j][l+1])**5))
                            
                Gpr[sum(n_list[:i]):sum(n_list[:i+1]),
                    sum(n_list[:j]):sum(n_list[:j+1])] = Gpr_block

    else:
        for i in xrange(M):

            # Compute p and r:
            RCi = R_list[i]*C_list[i]
            s = s_list[i]
            ts = ts_list[i]
            Gpr[n_sum, sum(n_list[:i]):sum(n_list[:i+1])] = \
                       Gpr[sum(n_list[:i]):sum(n_list[:i+1]), n_sum] = \
                       RCi*(1-exp(-s[1:]/RCi))
            Gpr[n_sum+1, sum(n_list[:i]):sum(n_list[:i+1])] = \
                         Gpr[sum(n_list[:i]):sum(n_list[:i+1]), n_sum+1] = \
                         RCi**2*((ts[1:]/RCi-1)-(ts[:-1]/RCi-1)*exp(-s[1:]/RCi))

            # Compute the quanta:
            qz[sum(n_list[:i]):sum(n_list[:i+1])] = \
                C_list[i]*d_list[i]-b_list[i]*RCi*(1-exp(-s[1:]/RCi))

            # Compute the G matrix:
            for j in xrange(M):
                Gpr_block = zeros((n_list[i], n_list[j]), float)
                RCj = R_list[j]*C_list[j]
                for k in xrange(n_list[i]):
                    for l in xrange(n_list[j]):
                        a1 = ts_list[i][k]
                        b1 = min(ts_list[j][l], ts_list[i][k+1])
                        a2 = max(ts_list[j][l], ts_list[i][k])
                        b2 = min(ts_list[j][l+1], ts_list[i][k+1])
                        a3 = max(ts_list[j][l+1], ts_list[i][k])
                        b3 = ts_list[i][k+1]

                        # The analytic expression for Gpr_block[k, l] is equivalent
                        # to the integration described in the comment below:
                        # f1 = lambda t: \
                        #      RCj**4*exp(-(ts_list[i][k+1]-t)/RCi)* \
                        #      (f((ts_list[j][l+1]-t)/RCj)-\
                        #      f((ts_list[j][l]-t)/RCj)*exp(-(ts_list[j][l+1]-ts_list[j][l])/RCj))
                        # f2 = lambda t: \
                        #      RCj**4*exp(-(ts_list[i][k+1]-t)/RCi)* \
                        #      (12*exp(-(ts_list[j][l+1]-t)/RCj)+ \
                        #      f((ts_list[j][l+1]-t)/RCj)+ \
                        #      f((ts_list[j][l]-t)/RCj)*exp(-(ts_list[j][l+1]-ts_list[j][l])/RCj))
                        # f3 = lambda t: \
                        #      RCj**4*exp(-(ts_list[i][k+1]-t)/RCi)* \
                        #      (f((ts_list[j][l]-t)/RCj)*exp(-(ts_list[j][l+1]-ts_list[j][l])/RCj)- \
                        #      f((ts_list[j][l+1]-t)/RCj))                              
                        # if (ts_list[i][k]<ts_list[j][l]):
                        #     Gpr_block[k, l] += quad(f1, a1, b1)[0]
                        # if (ts_list[j][l]<ts_list[i][k+1] and ts_list[j][l+1]>ts_list[i][k]):
                        #     Gpr_block[k, l] += quad(f2, a2, b2)[0]
                        # if (ts_list[j][l+1]<ts_list[i][k+1]):                        
                        #     Gpr_block[k, l] += quad(f3, a3, b3)[0]

                        F = lambda t, K, L, M, N: \
                            exp(-(M-t)/K)*(K*((N-t)/L)**3+(3*K**2/L-3*K)*((N-t)/L)**2+\
                                           (6*K**3/L**2-6*K**2/L+6*K)*((N-t)/L)+\
                                           (6*K**4/L**3-6*K**3/L**2+6*K**2/L-6*K))
                        F2 = lambda t, K, L, M, N: \
                             ((K*L)/(K+L))*exp(-(L*M+K*N-(K+L)*t)/(K*L))

                        if (ts_list[i][k]<ts_list[j][l]):
                            Gpr_block[k, l] += RCj**4*\
                                               ((F(b1, RCi, RCj, ts_list[i][k+1], ts_list[j][l+1])-\
                                                F(a1, RCi, RCj, ts_list[i][k+1], ts_list[j][l+1]))+\
                                                exp(-(ts_list[j][l+1]-ts_list[j][l])/RCj)*\
                                                (-F(b1, RCi, RCj, ts_list[i][k+1], ts_list[j][l])+\
                                                 F(a1, RCi, RCj, ts_list[i][k+1], ts_list[j][l])))
                        if (ts_list[j][l]<ts_list[i][k+1] and
                            ts_list[j][l+1]>ts_list[i][k]):
                            Gpr_block[k, l] += RCj**4*\
                                               (12*(F2(b2, RCi, RCj, ts_list[i][k+1], ts_list[j][l+1])-\
                                                    F2(a2, RCi, RCj, ts_list[i][k+1], ts_list[j][l+1]))+\
                                                F(b2, RCi, RCj, ts_list[i][k+1], ts_list[j][l+1])-\
                                                F(a2, RCi, RCj, ts_list[i][k+1], ts_list[j][l+1])+\
                                                exp(-(ts_list[j][l+1]-ts_list[j][l])/RCj)*\
                                                (F(b2, RCi, RCj, ts_list[i][k+1], ts_list[j][l])-\
                                                 F(a2, RCi, RCj, ts_list[i][k+1], ts_list[j][l])))
                        if (ts_list[j][l+1]<ts_list[i][k+1]):
                            Gpr_block[k, l] += RCj**4*\
                                               (exp(-(ts_list[j][l+1]-ts_list[j][l])/RCj)*\
                                                (F(b3, RCi, RCj, ts_list[i][k+1], ts_list[j][l])-\
                                                 F(a3, RCi, RCj, ts_list[i][k+1], ts_list[j][l]))-\
                                                F(b3, RCi, RCj, ts_list[i][k+1], ts_list[j][l+1])+\
                                                F(a3, RCi, RCj, ts_list[i][k+1], ts_list[j][l+1]))

                Gpr[sum(n_list[:i]):sum(n_list[:i+1]),
                    sum(n_list[:j]):sum(n_list[:j+1])] = Gpr_block
                
    # Compute the reconstruction coefficients:
    cd = dot(pinv(Gpr), qz)

    # Reconstruct the signal using the coefficients:
    t = arange(0, dur, dt)
    u_rec = cd[n_sum] + cd[n_sum+1]*t
    if all(isinf(R_list)):
        for j in xrange(M):
            ts = ts_list[j]
            psi = lambda t, k: \
                  0.25*where(t <= ts[k], ((t-ts[k+1])**4-(t-ts[k])**4),
                             where(t <= ts[k+1],
                                   ((t-ts[k+1])**4+(t-ts[k])**4),
                                   ((t-ts[k])**4-(t-ts[k+1])**4)))
            for k in xrange(n_list[j]):
                u_rec += cd[sum(n_list[:j])+k]*psi(t, k)                
    else:
        for j in xrange(M):
            RC = R_list[j]*C_list[j]
            ts = ts_list[j]
            psi = lambda t, k: \
                  (RC)**4*where(t <= ts[k],
                                f((ts[k+1]-t)/RC)-f((ts[k]-t)/RC)*exp(-(ts[k+1]-ts[k])/RC),
                                where(t <= ts[k+1],
                                      12*exp(-(ts[k+1]-t)/RC)+f((ts[k+1]-t)/RC)+
                                      f((ts[k]-t)/RC)*exp(-(ts[k+1]-ts[k])/RC),
                                      f((ts[k]-t)/RC)*exp(-(ts[k+1]-ts[k])/RC)-f((ts[k+1]-t)/RC)))
            for k in xrange(n_list[j]):
                u_rec += cd[sum(n_list[:j])+k]*psi(t, k)                

    return u_rec

def iaf_encode_coupled(u, dt, b_list, d_list, k_list, h_list, type_list):
    """
    Encode a signal with an ensemble of coupled ideal ON-OFF IAF
    neurons.

    Parameters
    ----------
    u : array_like of floats
        Signal to encode.
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is 1/dt Hz.
    b_list : list of floats
        List of encoder biases.
    d_list : list of floats
        List of encoder thresholds.
    k_list : list of floats
        List of encoder integration constants.
    h_list : M x M array_like of functions
        Coupling functions. Function `h_list[i][j]` describes the
        coupling from the integrator output of neuron `i` to the input
        of neuron `j`.
    type_list : list of integers {-1, 1}
        Neuron types. A value of -1 indicates that a neuron is an OFF-type
        neuron, while a value of 1 indicates that it is an ON-type
        neuron.

    Returns
    -------
    s_list : list of ndarrays of floats
        Encoded signal.
        
    """

    M = len(b_list)
    N = len(u)
    
    s_list = [[] for i in xrange(M)]
    ts_list = [[0.0] for i in xrange(M)]

    interval_list = [0.0 for i in xrange(M)]
    y_list = [0.0 for i in xrange(M)]
    for n in xrange(N):
        for i in xrange(M):

            # Rectangular quadrature is used to reduce the computational cost
            # of the integration:
            temp = u[n]+b_list[i]
            for j in xrange(M):
                ts = asarray(ts_list[j])
                temp += sum(h_list[j][i](n*dt-ts[ts<=ts_list[i][-1]]))
            y_list[i] += temp*dt/k_list[i]
            interval_list[i] += dt

            # Check whether the threshold was exceeded depending on
            # whether the neuron is an ON-type or OFF-type neuron:
            if (type_list[i] == 1 and y_list[i] >= d_list[i]) or \
                   (type_list[i] == -1 and y_list[i] <= d_list[i]):        
                s_list[i].append(interval_list[i])

                ## NOTE: computing and saving the spike time may
                ## potentially cause an overflow for very long signals:
                ts_list[i].append(interval_list[i]+ts_list[i][-1])
                interval_list[i] = 0.0
                y_list[i] = 0.0

    return [asarray(s) for s in s_list]

def iaf_decode_coupled(s_list, dur, dt, b_list, d_list, k_list, h_list):
    """
    Decode a signal encoded with an ensemble of coupled IAF ON-OFF
    neurons.

    Parameters
    ----------
    s_list : list of ndarrays of floats
        Signal encoded by an ensemble of coupled encoders. The values
        represent the time between spikes (in s). The number of arrays
        in the list corresponds to the number of encoders in the ensemble.
    dur : float
        Duration of signal (in s).
    dt : float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    b_list : list of floats
        List of encoder biases.
    d_list : list of floats
        List of encoder thresholds.
    k_list : list of floats
        List of encoder integration constants.
    h_list : M x M array_like of functions
        Coupling functions. Function `h_list[i][j]` describes the
        coupling from the integrator output of neuron `i` to the input
        of neuron `j`.
        
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

    # Compute the spike times:
    ts_list = map(cumsum, s_list)
    n_list = map(lambda ts: len(ts)-1, ts_list)

    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    n_sum = sum(n_list)
    Gpr = zeros((n_sum+2, n_sum+2), float)
    qz = zeros(n_sum+2, float)    

    for i in xrange(M):

        # Compute p and r:
        ts = ts_list[i]
        s = s_list[i][1:]
        Gpr[n_sum, sum(n_list[:i]):sum(n_list[:i+1])] = \
                   Gpr[sum(n_list[:i]):sum(n_list[:i+1]), n_sum] = \
                   ts[1:]-ts[:-1]
        Gpr[n_sum+1, sum(n_list[:i]):sum(n_list[:i+1])] = \
                     Gpr[sum(n_list[:i]):sum(n_list[:i+1]), n_sum+1] = \
                     (ts[1:]**2-ts[:-1]**2)/2

        # Compute the quanta:
        for k in xrange(n_list[i]):
            temp = k_list[i]*d_list[i]-b_list[i]*s[k]
            for j in xrange(M):
                for l in xrange(n_list[j]):
                    if ts_list[j][l] > ts[k]:
                        break
                    temp -= quad(lambda t: h_list[j][i](t-ts_list[j][l]), ts[k], ts[k+1])[0]

            qz[sum(n_list[:i])+k] = temp
        
        # Compute the G matrix:
        for j in xrange(M):
            Gpr_block = zeros((n_list[i], n_list[j]), float)
            for k in xrange(n_list[i]):
                for l in xrange(n_list[j]):
                    a1 = ts_list[i][k]
                    b1 = min(ts_list[j][l], ts_list[i][k+1])
                    a2 = max(ts_list[j][l], ts_list[i][k])
                    b2 = min(ts_list[j][l+1], ts_list[i][k+1])
                    a3 = max(ts_list[j][l+1], ts_list[i][k])
                    b3 = ts_list[i][k+1]
                        
                    if (ts_list[i][k]<ts_list[j][l]):
                        Gpr_block[k, l] += \
                                     0.05*(((b1-ts_list[j][l+1])**5-(b1-ts_list[j][l])**5)\
                                           -((a1-ts_list[j][l+1])**5-(a1-ts_list[j][l])**5))
                    if (ts_list[j][l]<ts_list[i][k+1] and ts_list[j][l+1]>ts_list[i][k]):
                        Gpr_block[k, l] += \
                                     0.05*(((b2-ts_list[j][l+1])**5+(b2-ts_list[j][l])**5)\
                                           -((a2-ts_list[j][l+1])**5+(a2-ts_list[j][l])**5))
                    if (ts_list[j][l+1]<ts_list[i][k+1]):
                        Gpr_block[k, l] += \
                                     0.05*(((b3-ts_list[j][l])**5-(b3-ts_list[j][l+1])**5)\
                                           -((a3-ts_list[j][l])**5-(a3-ts_list[j][l+1])**5))
                            
            Gpr[sum(n_list[:i]):sum(n_list[:i+1]),
                sum(n_list[:j]):sum(n_list[:j+1])] = Gpr_block

    cd = dot(pinv(Gpr), qz)

    # Reconstruct the signal using the coefficients:
    t = arange(0, dur, dt)
    u_rec = cd[n_sum] + cd[n_sum+1]*t
    for j in xrange(M):
        ts = ts_list[j]
        psi = lambda t, k: \
              0.25*where(t <= ts[k], ((t-ts[k+1])**4-(t-ts[k])**4),
                         where(t <= ts[k+1],
                               ((t-ts[k+1])**4+(t-ts[k])**4),
                               ((t-ts[k])**4-(t-ts[k+1])**4)))
        for k in xrange(n_list[j]):
            u_rec += cd[sum(n_list[:j])+k]*psi(t, k)                

    return u_rec

def iaf_encode_delay(u_list, T, dt, b_list, d_list, k_list, a_list,
                     w_list, y_list=None, interval_list=None,
                     u_list_prev=None, full_output=False):
    """
    Encode several signals with an ensemble of ideal IAF neurons with
    delays.

    Parameters
    ----------
    u_list : list of ndarrays of floats
        Signals to encode. Each of the ndarrays must be of the same
        length.
    T : float
        Temporal support of signals (in s). The portion of the signal
        encoded is `u_list[:][0:int(T/dt)]`.
    dt : float
        Sampling resolution of input signals; the sampling frequency
        is 1/dt Hz.
    b_list : list of floats
        List of encoder biases.
    d_list : list of floats
        List of encoder thresholds.
    k_list : list of floats
        List of encoder integration constants.
    a_list : N x M array_like of floats.
        Neuron delays (in s).
    w_list : N x M array_like of floats.
        Scaling factors.
    y_list : list of floats
        Initial values of integrators.
    interval_list : list of floats
        Times since last spikes (in s).
    u_list_prev : list of ndarrays of floats
        If nonempty, the contents of this list are prepended to the
        contents of `u_list`.
    full_output : bool
        If set, the function returns the encoded data block followed
        by the given parameters (with updated values for `y` and `interval`).
        This is useful when the function is called repeatedly to
        encode a long signal.

    Returns
    -------
    s_list : list of ndarrays of floats
        If `full_output` == False, returns the signals encoded as a list
        of arrays of time intervals between spikes.
    s_list, T, dt, b_list, d_list, k_list, a_list, w_list, y_list,
    interval_list, u_list_prev, full_output : tuple
        If `full_output` == True, returns the encoded signals followed
        by updated encoder parameters.

    Notes
    -----
    The specified signal length, i.e., `len(u_list[0])*dt`, must
    exceed the support `T` over which the signal is encoded by the
    length of the longest delay. 
    
    """

    M = len(u_list) # number of input signals
    if not M:
        raise ValueError('no spike data given')

    if u_list_prev != None:
        if len(u_list_prev) != M:
            raise ValueError('u_list_prev must have the same number ' +
                             'of entries as u_list')                   
        u_list = [hstack((u_prev, u)) for u_prev, u in
                  zip(u_list_prev, u_list)]
    if len(set(map(len, u_list))) > 1:
        raise ValueError('all input signals must be of the same length')
    N = len(b_list) # number of neurons
    if shape(a_list) != (N, M):
        raise ValueError('incorrect number of delay parameters')
    if shape(w_list) != (N, M):
        raise ValueError('incorrect number of scaling factors')
    
    a_max = amax(a_list)
    if len(u_list[0])*dt < T+a_max:
        raise ValueError('signals insufficiently long')

    s_list = [[] for i in xrange(N)]
    if interval_list == None:
        interval_list = [0.0 for i in xrange(N)]
    if y_list == None:        
        y_list = [0.0 for i in xrange(N)]
    for j in xrange(N):

        # Rectangular quadrature is used to reduce the computational
        # cost of the integration:
        for k in xrange(int(T/dt)):
            v = 0.0

            # The portion of the signal encoded begins at time
            # len(u_list[0])*dt-T:
            for i in xrange(M):
                u_delayed = u_list[i][int(a_list[j][i]/dt):int((T+a_list[j][i])/dt)]
                v += w_list[j][i]*u_delayed[k]
            y_list[j] += dt*(v+b_list[j])/k_list[j]
            interval_list[j] += dt

            # Generate a spike when the threshold is exceeded:
            if y_list[j] >= d_list[j]:
                s_list[j].append(interval_list[j])
                interval_list[j] = 0.0
                y_list[j] = 0.0

    # u_list_prev is set to contain the values in u_list that occur
    # after time T:
    u_list_prev = [u[int(T/dt):] for u in u_list]
    
    if full_output:
        return [[asarray(s) for s in s_list], T, dt, b_list, d_list, \
               k_list, a_list, w_list, y_list, interval_list, \
               u_list_prev, full_output]
    else:        
        return [asarray(s) for s in s_list]

def iaf_decode_delay(s_list, T, dt, b_list, d_list, k_list, a_list, w_list):
    """
    Decode several signals encoded with an ensemble of ideal IAF
    neurons with delays.

    Parameters
    ----------
    s_list : list of ndarrays of floats
        Signals encoded by an ensemble of encoders. The values
        represent the time between spikes (in s). The number of arrays
        in the list corresponds to the number of encoders in the ensemble.
    T : float
        Temporal support of signals (in s). 
    dt : float
        Sampling resolution of input signals; the sampling frequency
        is 1/dt Hz.
    b_list : list of floats
        List of encoder biases.
    d_list : list of floats
        List of encoder thresholds.
    k_list : list of floats
        List of encoder integration constants.
    a_list : N x M array_like of floats.
        Delays (in s).
    w_list : N x M array_like of floats.
        Scaling factors.

    Returns
    -------
    u_list : list of ndarrays of floats
        Decoded signals.

    Notes
    -----
    The specified signal length must exceed the support `T` over which
    the signal is decoded by the length of the longest delay.
    
    """

    N = len(s_list)      # number of neurons
    M = shape(a_list)[1] # number of decoded signals
    
    # Compute the spike times:
    ts_list = map(cumsum, s_list)
    n_list = map(lambda ts: len(ts)-1, ts_list)

    # Compute the values of the matrix that must be inverted to obtain
    # the reconstruction coefficients:
    n_sum = sum(n_list)
    Gpr = zeros((n_sum+2*M, n_sum+2*M), float)
    qz = zeros(n_sum+2*M, float)
    for j in xrange(N):

        # Compute the quanta:
        qz[sum(n_list[:j]):sum(n_list[:j+1])] = \
            k_list[j]*d_list[j]-b_list[j]*s_list[j][1:]

        # Compute p and r:
        for i in xrange(M):
            tau = ts_list[j] + a_list[j][i]
            w = w_list[j][i]
            p = zeros(n_list[j], float)
            r = zeros(n_list[j], float)
            for k in xrange(n_list[j]):
                if tau[k+1] < T:
                     p[k] = w*(tau[k+1]-tau[k])
                     r[k] = 0.5*w*(tau[k+1]**2-tau[k]**2)
                elif tau[k] < T:
                     p[k] = w*(T-tau[k])
                     r[k] = 0.5*w*(T**2-tau[k]**2)
                else:
                     p[k] = 0.0
                     r[k] = 0.0
            Gpr[sum(n_list[:j]):sum(n_list[:j+1]), n_sum+i] = \
                Gpr[n_sum+i, sum(n_list[:j]):sum(n_list[:j+1])] = p
            Gpr[sum(n_list[:j]):sum(n_list[:j+1]), n_sum+i+M] = \
                Gpr[n_sum+i+M, sum(n_list[:j]):sum(n_list[:j+1])] = r

    for i in xrange(N):                
        for j in xrange(N):

            # Compute the G matrix:
            Gpr_block = zeros((n_list[i], n_list[j]), float)
            for k in xrange(n_list[i]):
                for l in xrange(n_list[j]):
                    for m in xrange(M):
                        tau_im = ts_list[i]+a_list[i][m]
                        tau_jm = ts_list[j]+a_list[j][m]

                        # The analytic expression for Gpr_block[k, l]
                        # is equivalent to the integration described
                        # in the comment below.
                        # def psi(t):
                        #     if t <= tau_jm[l]:
                        #         result = (t-tau_jm[l+1])**4-(t-tau_jm[l])**4
                        #     elif t <= tau_jm[l+1]:
                        #         result = (t-tau_jm[l+1])**4+(t-tau_jm[l])**4
                        #     else:
                        #         result = (t-tau_jm[l])**4-(t-tau_jm[l+1])**4
                        #     return w_list[j][m]*0.25*result
                        # Gpr_block[k, l] += \
                        #              w_list[i][m]*quad(psi, tau_im[k],
                        #                                tau_im[k+1])[0]
                        
                        temp = 0.0
                        if tau_jm[l+1] <= tau_im[k]:
                            temp += \
                                 (tau_im[k+1]-tau_jm[l])**5+(tau_im[k]-tau_jm[l+1])**5-\
                                 (tau_im[k]-tau_jm[l])**5-(tau_im[k+1]-tau_jm[l+1])**5
                        if (tau_jm[l] <= tau_im[k]) and (tau_im[k] <= tau_jm[l+1]) and \
                               (tau_jm[l+1] <= tau_im[k+1]):
                            temp += \
                                 (tau_im[k+1]-tau_jm[l])**5-(tau_im[k]-tau_jm[l+1])**5-\
                                 (tau_im[k]-tau_jm[l])**5-(tau_im[k+1]-tau_jm[l+1])**5
                        if (tau_jm[l] <= tau_im[k]) and (tau_im[k] <= tau_im[k+1]) and \
                               (tau_im[k+1] <= tau_jm[l+1]):
                            temp += \
                                 (tau_im[k+1]-tau_jm[l])**5-(tau_im[k]-tau_jm[l+1])**5-\
                                 (tau_im[k]-tau_jm[l])**5+(tau_im[k+1]-tau_jm[l+1])**5        
                        if (tau_im[k] <= tau_jm[l]) and (tau_jm[l] <= tau_jm[l+1]) and \
                               (tau_jm[l+1] <= tau_im[k+1]):
                            temp += \
                                 (tau_im[k+1]-tau_jm[l])**5-(tau_im[k]-tau_jm[l+1])**5+\
                                 (tau_im[k]-tau_jm[l])**5-(tau_im[k+1]-tau_jm[l+1])**5
                        if (tau_im[k] <= tau_jm[l]) and (tau_jm[l] <= tau_im[k+1]) and \
                               (tau_im[k+1] <= tau_jm[l+1]):
                            temp += \
                                 (tau_im[k+1]-tau_jm[l])**5-(tau_im[k]-tau_jm[l+1])**5+\
                                 (tau_im[k]-tau_jm[l])**5+(tau_im[k+1]-tau_jm[l+1])**5
                        if tau_im[k+1] <= tau_jm[l]:
                            temp += \
                                 -(tau_im[k+1]-tau_jm[l])**5-(tau_im[k]-tau_jm[l+1])**5+\
                                 (tau_im[k]-tau_jm[l])**5+(tau_im[k+1]-tau_jm[l+1])**5
                        Gpr_block[k, l] += temp*w_list[i][m]*w_list[j][m]/20.0
            
            Gpr[sum(n_list[:i]):sum(n_list[:i+1]),
                sum(n_list[:j]):sum(n_list[:j+1])] = Gpr_block

    # Compute the reconstruction coefficients:
    cd = dot(pinv(Gpr), qz)

    # Reconstruct the signal over the specified support using the
    # coefficients:
    t = arange(0, T, dt)
    u_rec_list = []
    for i in xrange(M):
        u_rec_list.append(cd[n_sum+i]+cd[n_sum+i+M]*t)
        for j in xrange(N):
            tau = ts_list[j]+a_list[j][i]
            psi = lambda t, k: \
                  0.25*w_list[j][i]*where(t <= tau[k],
                                          ((t-tau[k+1])**4-(t-tau[k])**4),
                                          where(t <= tau[k+1],
                                                ((t-tau[k+1])**4+(t-tau[k])**4),
                                                ((t-tau[k])**4-(t-tau[k+1])**4)))

            # Compute offset before loop to save time:
            nj = sum(n_list[:j])
            for k in xrange(n_list[j]):
                u_rec_list[i] += cd[nj+k]*psi(t, k)
            
    return u_rec_list 

