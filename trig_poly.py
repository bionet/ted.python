#!/usr/bin/env python

"""
Routines for manipulating trigonometric polynomials.
"""

import numpy as np

def em(m, t, Omega, M):
    """
    Trigonometric polynomial basis function.

    Parameters
    ----------
    m : int
        Order of basis function.
    t : ndarray
        Times over which to compute the function.
    Omega : float
        Bandwidth (in rad/s).

    Returns
    -------
    u : ndarray
        Basis function `exp(1j*m*Omega*t/M)/sqrt(TM)`, where
        `TM == 2*pi*M/Omega`.
    
    """

    TM = 2*np.pi*M/Omega
    return np.exp(1j*m*Omega*t/M)/np.sqrt(TM)

def gen_trig_poly_coeffs(M):
    """
    Generate random Dirichlet coefficients for a real signal.

    Parameters
    ----------
    M : int
        `2*M+1` complex coefficients are generated.

    Returns
    -------
    am : ndarray
        An array containing the coefficients.
        
    """
    
    am = np.empty(2*M+1, np.complex)
    am[0:M] = np.random.rand(M)+1j*np.random.rand(M)
    am[M] = np.random.rand(1)
    am[-1:-M-1:-1] = np.conj(am[0:M])

    return am

def gen_trig_poly(t, Omega, am):
    """
    Construct a trigonometric polynomial with specified Dirichlet coefficients.

    Parameters
    ----------
    t : ndarray
        Times over which to compute the trigonometric polynomial.
    Omega : float
        Bandwidth of signal (in rad/s).
    am : ndarray
        Dirichlet coefficients. The length of this array must be odd.

    Returns
    -------
    u : ndarray
        Generated signal.
        
    """

    N = len(am)
    M = int(np.floor(N/2))
    if M < 1:
        raise ValueError('number of coefficients must be at least 1')
    if not (N % 2):
        raise ValueError('number of coefficients must be odd')

    u = np.zeros(len(t), np.complex)
    for m in xrange(-M, M+1):
        u += am[m+M]*em(m, t, Omega, M)

    return np.real(u)

def gen_trig_poly_fft(t, Omega, am):
    """
    Construct a trigonometric polynomial with specified Dirichlet coefficients.

    Parameters
    ----------
    t : ndarray
        Times over which to compute the trigonometric polynomial.
    Omega : float
        Bandwidth of signal (in rad/s).
    am : ndarray
        Dirichlet coefficients. The length of this array must be odd.

    Returns
    -------
    u : ndarray
        Generated signal.

    Notes
    -----
    This function uses the FFT to generate the output signal.
    
    """

    N = len(am)
    M = N/2
    if M < 1:
        raise ValueError('number of coefficients must be at least 1')
    if N % 2 == 0: 
        raise ValueError('number of coefficients must be odd')

    dt = t[1]-t[0]
    TM = 2*np.pi*M/Omega
    u_fft = np.zeros(len(t), np.complex)
    u_fft[len(t)-M:] = am[0:M]*np.sqrt(TM)/dt
    u_fft[0:M+1] = am[M:]*np.sqrt(TM)/dt

    return np.real(np.fft.ifft(u_fft))

def get_dir_coeff_inner(u, dt, Omega, M):
    """
    Compute the Dirichlet coefficients of a trigonometric polynomial
    using inner products.

    Notes
    -----
    Assumes that `u` is defined over times `dt*arange(0, len(t))` and
    that `dt*len(u)` is equal to the period of the trigonometric polynomial.

    """

    t = np.arange(0, len(u))*dt
    am = np.empty(2*M+1, np.complex)
    TM = 2*np.pi*M/Omega
    for m in xrange(-M, M+1):
        am[m+M] = dt*np.sum(u*em(-m, t, Omega, M))
    return am

def get_dir_coeff_fft(u, dt, Omega, M):
    """
    Compute the Dirichlet coefficients of a trigonometric polynomial
    using the FFT.

    Notes
    -----
    Assumes that `u` is defined over times `dt*arange(0, len(t))` and
    that `dt*len(u)` is equal to the period of the trigonometric polynomial.    

    """

    TM = 2*np.pi*M/Omega
    u_fft = np.fft.fft(u)
    am = np.empty(2*M+1, np.complex)
    am[0:M] = u_fft[-M:]*dt/np.sqrt(TM)
    am[M:2*M+1] = u_fft[0:M+1]*dt/np.sqrt(TM)
    
    return am

def filter_trig_poly(u, h, dt, Omega, M):
    """
    Filter a trigonometric signal with a filter's impulse response.

    Parameters
    ----------
    u : ndarray
        Input signal.
    h : ndarray
        Impulse response of filter.
    dt : float

    Returns
    -------
    v : ndarray
        Filtered signal.
        
    Notes
    -----
    Assumes that `u` is defined over times `dt*arange(0, len(t))` and
    that `dt*len(u)` is equal to the period of the trigonometric polynomial.    

    """

    # Get the Dirichlet coefficients of the signal:
    am = get_dir_coeff_fft(u, dt, Omega, M)

    # Get the Dirichlet coefficients of the filter:
    hm = get_dir_coeff_fft(h, dt, Omega, M)

    # Construct the filtered signal using the above coefficients:
    t = np.arange(0, len(u))*dt
    v = np.zeros(len(u), np.complex)
    TM = 2*np.pi*M/Omega
    for m in xrange(-M, M+1):
        v += hm[m+M]*am[m+M]*em(m, t, Omega, M)*np.sqrt(TM)/dt

    return np.real(v)

def filter_trig_poly_fft(u, h):
    """
    Filter a trigonometric signal with a filter's impulse response.

    Parameters
    ----------
    u : ndarray
        Input signal.
    h : ndarray
        Impulse response of filter.
    dt : float

    Returns
    -------
    v : ndarray
        Filtered signal.
        
    Notes
    -----
    Assumes that `u` is defined over times `dt*arange(0, len(t))` and
    that `dt*len(u)` is equal to the period of the trigonometric polynomial.    

    This function uses the FFT to perform the filtering.
    
    """

    N = len(u)
    return np.real(np.fft.ifft(np.fft.fft(np.hstack((u, np.zeros(len(h)-1))))*\
                               np.fft.fft(np.hstack((h, np.zeros(len(u)-1))))))[0:N]

