#!/usr/bin/env python

"""
Signal Processing Extras
========================

This module contains various signal processing tools and
algorithms that are not currently in scipy.signal.

Error Analysis Routines
-----------------------
- db              Convert a power value to decibels.
- rms             Compute the root mean squared value of an array.
- snr             Compute the signal-to-noise ratio of two signals.

Filtering Routines
------------------
- downsample      Downsample an array.
- fftfilt         Apply an FIR filter to a signal using the overlap-add method.
- remezord        Determine filter parameters for Remez algorithm.
- upsample        Upsample an array.

Miscellaneous Routines
----------------------
- nextpow2        Return n such that 2**n >= abs(x).
- oddceil         Return the smallest odd integer no less than x.
- oddround        Return the nearest odd integer nearest to x.

"""

__all__ = ['db', 'downsample', 'fftfilt', 'nextpow2', 'oddceil', 'oddround',
           'remezord', 'rms', 'snr', 'upsample']

from numpy import abs, arange, arctan, argmin, asarray, ceil, floor, \
     hstack, int, log10, log2, max, mean, min, mod, pi, shape, \
     sqrt, zeros

# Since the fft function in scipy is faster than that in numpy, try to
# import the former before falling back to the latter:
try:
    from scipy.fftpack import fft, ifft
except ImportError:
    from numpy.fft import fft, ifft

# --- Error analysis functions ---

def db(x):
    """Convert the specified power value to decibels assuming a
    reference value of 1."""

    return 10*log10(x)

def rms(x):
    """Compute the root mean squared value of the specified array x."""

    return sqrt(mean(abs(x)**2))

def snr(u, u_rec, k_min=0, k_max=None):
    """Compute the signal-to-noise ratio (in dB) of a signal given its
    reconstruction.

    Parameters
    ----------
    u : numpy array of floats
        Original signal.
    u_rec : numpy array of floats
        Reconstructed signal.
    k_min : int
        Lower index into the signal over which to compute the SNR.
    k_max : int
        Upper index into the signal over which to compute the SNR.

    """

    if len(u) != len(u_rec):
        raise ValueError('u and u_rec must be the same length')

    return db(mean(u[k_min:k_max]**2))-db(mean((u[k_min:k_max]-u_rec[k_min:k_max])**2))

# --- Sampling functions ---

def upsample(x, n, offset=0):
    """Upsample a vector x by inserting n-1 zeros between every
    entry. An optional offset may be specified."""

    if len(shape(x)) > 1:
        raise ValueError('x must be a vector')
    y = zeros(len(x)*n, asarray(x).dtype)
    y[offset::n] = x
    return y

def downsample(x, n, offset=0):
    """Downsample a vector x by returning every nth entry. An optional
    offset may be specified."""

    if len(shape(x)) > 1:
        raise ValueError('x must be a vector')
    return x[offset::n]

# --- Filtering functions ---

def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""

    return ceil(log2(abs(x)))

def fftfilt(b, x, *n):
    """Filter the signal x with the FIR filter described by the
    coefficients in b using the overlap-add method. If the FFT
    length n is not specified, it and the overlap-add block length
    are selected so as to minimize the computational cost of
    the filtering operation."""

    N_x = len(x)
    N_b = len(b)

    # Determine the FFT length to use:
    if len(n):

        # Use the specified FFT length (rounded up to the nearest
        # power of 2), provided that it is no less than the filter
        # length:
        n = n[0]
        if n != int(n) or n <= 0:
            raise ValueError('n must be a nonnegative integer')
        if n < N_b:
            n = N_b
        N_fft = 2**nextpow2(n)
    else:

        if N_x > N_b:

            # When the filter length is smaller than the signal,
            # choose the FFT length and block size that minimize the
            # FLOPS cost. Since the cost for a length-N FFT is
            # (N/2)*log2(N) and the filtering operation of each block
            # involves 2 FFT operations and N multiplications, the
            # cost of the overlap-add method for 1 length-N block is
            # N*(1+log2(N)). For the sake of efficiency, only FFT
            # lengths that are powers of 2 are considered:
            N = 2**arange(ceil(log2(N_b)), floor(log2(N_x)))
            cost = ceil(N_x/(N-N_b+1))*N*(log2(N)+1)
            N_fft = N[argmin(cost)]

        else:

            # When the filter length is at least as long as the signal,
            # filter the signal using a single block:
            N_fft = 2**nextpow2(N_b+N_x-1)

    N_fft = int(N_fft)

    # Compute the block length:
    L = int(N_fft - N_b + 1)

    # Compute the transform of the filter:
    H = fft(b, N_fft)

    y = zeros(N_x,float)
    i = 0
    while i <= N_x:
        il = min([i+L,N_x])
        k = min([i+N_fft,N_x])
        yt = ifft(fft(x[i:il],N_fft)*H,N_fft) # Overlap..
        y[i:k] = y[i:k] + yt[:k-i]            # and add
        i += L
    return y

def oddround(x):
    """Return the nearest odd integer from x."""

    return x-mod(x,2)+1

def oddceil(x):
    """Return the smallest odd integer no less than x."""

    return oddround(x+1)

def remlplen_herrmann(fp, fs, dp, ds):
    """Determine the length of the low pass filter with passband frequency
    fp, stopband frequency fs, passband ripple dp, and stopband ripple ds.
    fp and fs must be normalized with respect to the sampling frequency.
    Note that the filter order is one less than the filter length.

    References
    ----------
    O. Herrmann, L.R. Raviner, and D.S.K. Chan, Practical Design Rules for
    Optimum Finite Impulse Response Low-Pass Digital Filters, Bell Syst. Tech.
    Jour., 52(6):769-799, Jul./Aug. 1973.

    """

    dF = fs-fp
    a = [5.309e-3,7.114e-2,-4.761e-1,-2.66e-3,-5.941e-1,-4.278e-1]
    b = [11.01217, 0.51244]
    Dinf = log10(ds)*(a[0]*log10(dp)**2+a[1]*log10(dp)+a[2])+ \
           a[3]*log10(dp)**2+a[4]*log10(dp)+a[5]
    f = b[0]+b[1]*(log10(dp)-log10(ds))
    N1 = Dinf/dF-f*dF+1

    return int(oddround(N1))

def remlplen_kaiser(fp, fs, dp, ds):
    """Determine the length of the low pass filter with passband frequency
    fp, stopband frequency fs, passband ripple dp, and stopband ripple ds.
    fp and fs must be normalized with respect to the sampling frequency.
    Note that the filter order is one less than the filter length.

    References
    ----------
    J.F. Kaiser, Nonrecursive Digital Filter Design Using I_0-sinh Window
    function, Proc. IEEE Int. Symp. Circuits and Systems, 20-23, April 1974.

    """

    dF = fs-fp
    N2 = (-20*log10(sqrt(dp*ds))-13.0)/(14.6*dF)+1.0

    return int(oddceil(N2))

def remlplen_ichige(fp, fs, dp, ds):
    """Determine the length of the low pass filter with passband frequency
    fp, stopband frequency fs, passband ripple dp, and stopband ripple ds.
    fp and fs must be normalized with respect to the sampling frequency.
    Note that the filter order is one less than the filter length.

    References
    ----------
    K. Ichige, M. Iwaki, and R. Ishii, Accurate Estimation of Minimum
    Filter Length for Optimum FIR Digital Filters, IEEE Transactions on
    Circuits and Systems, 47(10):1008-1017, October 2000.

    """
    
    dF = fs-fp
    v = lambda dF,dp:2.325*((-log10(dp))**-0.445)*dF**(-1.39)
    g = lambda fp,dF,d:(2.0/pi)*arctan(v(dF,dp)*(1.0/fp-1.0/(0.5-dF)))
    h = lambda fp,dF,c:(2.0/pi)*arctan((c/dF)*(1.0/fp-1.0/(0.5-dF)))
    Nc = ceil(1.0+(1.101/dF)*(-log10(2.0*dp))**1.1)
    Nm = (0.52/dF)*log10(dp/ds)*(-log10(dp))**0.17
    N3 = ceil(Nc*(g(fp,dF,dp)+g(0.5-dF-fp,dF,dp)+1.0)/3.0)
    DN = ceil(Nm*(h(fp,dF,1.1)-(h(0.5-dF-fp,dF,0.29)-1.0)/2.0))
    N4 = N3+DN

    return int(N4)

def remezord(freqs, amps, rips, Hz=1, alg='ichige'):
    """Calculate the parameters required by the Remez exchange algorithm to
    construct a finite impulse response (FIR) filter that approximately
    meets the specified design.

    Parameters
    ----------
    freqs : array_like of floats
        A monotonic sequence of band edges specified in Hertz. All
        elements must be non-negative and less than 1/2 the
        sampling frequency as given by the Hz parameter.
    amps : array_like of floats
        A sequence containing the amplitudes of the signal to be
        filtered over the various bands.
    rips : array_like of floats
        A sequence specifying the maximum ripples of each band.
    alg : {'herrmann', 'kaiser', 'ichige'}
        Filter length approximation algorithm.

    Returns
    -------
    numtaps : int
        Desired number of filter taps.
    bands : ndarray of floats
        A monotonic sequence containing the band edges.
    amps : ndarray of floats
        Desired gain for each band region.
    weights : ndarray of floats
        Filter design weights.

    See Also
    --------
    scipy.signal.remez

    """

    # Make sure the parameters are floating point numpy arrays:
    freqs = asarray(freqs, 'd')
    amps = asarray(amps, 'd')
    rips = asarray(rips, 'd')

    # Scale ripples with respect to band amplitudes:
    rips /= (amps+(amps==0.0))

    # Normalize input frequencies with respect to sampling frequency:
    freqs /= Hz

    # Select filter length approximation algorithm:
    if alg == 'herrmann':
        remlplen = remlplen_herrmann
    elif alg == 'kaiser':
        remlplen = remlplen_kaiser
    elif alg == 'ichige':
        remlplen = remlplen_ichige
    else:
        raise ValueError('Unknown filter length approximation algorithm.')

    # Validate inputs:
    if any(freqs > 0.5):
        raise ValueError('Frequency band edges must not exceed the Nyquist frequency.')
    if any(freqs < 0.0):
        raise ValueError('Frequency band edges must be nonnegative.')
    if any(rips < 0.0):
        raise ValueError('Ripples must be nonnegative.')
    if len(amps) != len(rips):
        raise ValueError('Number of amplitudes must equal number of ripples.')
    if len(freqs) != 2*(len(amps)-1):
        raise ValueError('Number of band edges must equal 2*((number of amplitudes)-1)')

    # Find the longest filter length needed to implement any of the
    # low-pass or high-pass filters with the specified edges:
    f1 = freqs[0:-1:2]
    f2 = freqs[1::2]
    L = 0
    for i in range(len(amps)-1):
        L = max((L,
                 remlplen(f1[i], f2[i], rips[i], rips[i+1]),
                 remlplen(0.5-f2[i], 0.5-f1[i], rips[i+1], rips[i])))

    # Cap the sequence of band edges with the limits of the digital frequency
    # range:
    bands = hstack((0.0, freqs, 0.5))

    # The filter design weights correspond to the ratios between the maximum
    # ripple and all of the other ripples:
    weight = max(rips)/rips

    return [L, bands, amps, weight]
