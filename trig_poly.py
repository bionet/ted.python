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
    t : numpy.ndarray
        Times over which to compute the function.
    Omega : float
        Bandwidth (in rad/s).
    M : int
        Trigonometric polynomial order.
        
    Returns
    -------
    u : numpy.ndarray
        Basis function `exp(1j*m*Omega*t/M)/sqrt(T)`, where
        `T == 2*pi*M/Omega`.
    
    """

    T = 2*np.pi*M/Omega
    return np.exp(1j*m*Omega*t/M)/np.sqrt(T)

def gen_dirichlet_coeffs(M):
    """
    Generate random Dirichlet coefficients for a real signal.

    Parameters
    ----------
    M : int
        Trigonometric polynomial order.

    Returns
    -------
    am : numpy.ndarray
        Array of `2*M+1` Dirichlet coefficients.
        
    """
    
    am = np.empty(2*M+1, np.complex)
    am[0:M] = np.random.rand(M)+1j*np.random.rand(M)
    am[M] = np.random.rand(1)
    am[-1:-M-1:-1] = np.conj(am[0:M])

    return am

def gen_trig_poly(t, am):
    """
    Construct a trigonometric polynomial with specified Dirichlet coefficients.

    Parameters
    ----------
    t : numpy.ndarray
        Times over which to compute the trigonometric polynomial.
    am : numpy.ndarray
        Dirichlet coefficients. The length of this array must be odd.

    Returns
    -------
    u : numpy.ndarray
        Generated signal.
        
    """

    N = len(am)
    M = int(np.floor(N/2))
    if M < 1:
        raise ValueError('number of coefficients must be at least 1')
    if not (N % 2):
        raise ValueError('number of coefficients must be odd')

    T = max(t)-min(t)
    Omega = 2*np.pi*M/T
    
    u = np.zeros(len(t), np.complex)
    for m in xrange(-M, M+1):
        u += am[m+M]*em(m, t, Omega, M)

    return np.real(u)

def gen_trig_poly_fft(t, am):
    """
    Construct a trigonometric polynomial with specified Dirichlet coefficients.

    Parameters
    ----------
    t : numpy.ndarray
        Times over which to compute the trigonometric polynomial.
    am : numpy.ndarray
        Dirichlet coefficients. The length of this array must be odd.

    Returns
    -------
    u : numpy.ndarray
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
    T = max(t)-min(t)
    Omega = 2*np.pi*M/T
    u_fft = np.zeros(len(t), np.complex)
    u_fft[len(t)-M:] = am[0:M]*np.sqrt(T)/dt
    u_fft[0:M+1] = am[M:]*np.sqrt(T)/dt

    return np.real(np.fft.ifft(u_fft))

def get_dirichlet_coeffs_inner(u, dt, M):
    """
    Compute the Dirichlet coefficients of a trigonometric polynomial.

    Parameters
    ----------
    u : numpy.ndarray
        Input signla.
    dt : float
        Time resolution (s).
    M : int
        Trigonometric polynomial order.

    Returns
    -------
    am : numpy.ndarray
        Array of `2*M+1` Dirichlet coefficients.
        
    Notes
    -----    
    Assumes that `u` is defined over times `dt*arange(0, len(u))` and
    that `dt*len(u)` is equal to the period of the trigonometric polynomial.

    This routine computes the coefficients using inner products.
    
    """

    T = dt*len(u)
    Omega = 2*np.pi*M/T
    t = np.arange(0, len(u))*dt
    am = np.empty(2*M+1, np.complex)
    for m in xrange(-M, M+1):
        am[m+M] = dt*np.sum(u*em(-m, t, Omega, M))
    return am

def get_dirichlet_coeffs_fft(u, dt, M):
    """
    Compute the Dirichlet coefficients of a trigonometric polynomial.

    Parameters
    ----------
    u : numpy.ndarray
        Input signla.
    dt : float
        Time resolution (s).
    M : int
        Trigonometric polynomial order.

    Returns
    -------
    am : numpy.ndarray
        Array of `2*M+1` Dirichlet coefficients.

    Notes
    -----
    Assumes that `u` is defined over times `dt*arange(0, len(u))` and
    that `dt*len(u)` is equal to the period of the trigonometric polynomial.    

    This routine computes the coefficients using the FFT.
    
    """

    T = dt*len(u)
    Omega = 2*np.pi*M/T
    u_fft = np.fft.fft(u)
    am = np.empty(2*M+1, np.complex)
    am[0:M] = u_fft[-M:]*dt/np.sqrt(T)
    am[M:2*M+1] = u_fft[0:M+1]*dt/np.sqrt(T)
    
    return am


