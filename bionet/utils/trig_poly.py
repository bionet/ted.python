#!/usr/bin/env python

"""
Routines for Manipulating Trigonometric Polynomials
===================================================
- em                    Trigonometric polynomial basis function.
- gen_dirichlet_coeffs  Generate random Dirichlet coefficients.
- gen_trig_poly         Generate a 1D trigonometric polynomial.
- gen_trig_poly_2d      Generate a 2D trigonometric polynomial.
- get_dirichlet_coeffs  Compute Dirichlet coefficients of a signal.
- scale_down_coeffs     Scale down Dirichlet coefficients.
"""

# Copyright (c) 2009-2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

__all__ = ['em', 'scale_down_coeffs', 'gen_dirichlet_coeffs',
           'gen_trig_poly', 'get_dirichlet_coeffs',
           'gen_trig_poly_2d']

import numpy as np

from numpy_extras import crand

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

def scale_down_coeffs(am):
    """
    Scale down Dirichlet coefficients.

    Parameters
    ----------
    am : numpy.ndarray
        Array of Dirichlet coefficients of a real trigonometric
        polynomial. Must be of odd length.

    Returns
    -------
    am_new : numpy.ndarray
        Array of scaled coefficients; the highest frequency
        coefficients are scaled down the most.
        
    """

    if len(am) % 2 == 0:
        raise ValueError('array length must be odd')
    M = len(am)/2
    am_new = np.copy(am)
    am_new[0:M] *= np.arange(1.0, M+1)/M
    am_new[M+1:] *= np.arange(M, 0.0, -1)/M 
    return am_new

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
        Array of Dirichlet coefficients with shape `(2*M+1)`. The
        coefficients are ordered such that `am[0]` contains the
        coefficient for `m == -M`.
        
    """
    
    am = np.empty(2*M+1, np.complex)
    am[0:M] = np.random.rand(M)+1j*np.random.rand(M)
    am[M] = np.random.rand(1)
    am[-1:-M-1:-1] = np.conj(am[0:M])
    return am

def gen_trig_poly(T, dt, am, method='fft', scale_down=False):
    """
    Construct a trigonometric polynomial with specified Dirichlet coefficients.

    Parameters
    ----------
    T : float
        Period (i.e., duration) of the trigonometric polynomial.
    dt : float
        Time resolution.
    am : int or numpy.ndarray
        Trigonometric polynomial order or array of Dirichlet
        coefficients. If the latter, the length of the array must be odd.
    method : {'fft', 'inner'}
        Method to use when computing coefficients. The FFT method is
        generally faster than using inner products.
    scale_down : bool
        If true, linearly scale down all coefficients such that the
        magnitude of the high-frequency coefficients are reduced the most.
        
    Returns
    -------
    u : numpy.ndarray
        Generated signal.
        
    """

    if isinstance(am, int):
        M = am
        am = gen_dirichlet_coeffs(M)
    elif np.iterable(am):
        if len(am) % 2 == 0:
            raise ValueError('number of coefficients must be odd')
        M = len(am)/2
    else:
        raise ValueError('unrecognized parameter type')
    if M < 1:
        raise ValueError('number of coefficients must be at least 1')

    if scale_down:
        am = scale_down_coeffs(am)
        
    N = int(np.ceil(T/dt))
    if method == 'fft':
        u_fft = np.zeros(N, np.complex)
        u_fft[N-M:] = am[0:M]*np.sqrt(T)/dt
        u_fft[0:M+1] = am[M:]*np.sqrt(T)/dt
        return np.real(np.fft.ifft(u_fft))

    elif method == 'inner':
        t = np.arange(0, T, dt)
        u = np.zeros(N, np.complex)
        Omega = 2*np.pi*M/T
        for m in xrange(-M, M+1):
            u += am[m+M]*em(m, t, Omega, M)
        return np.real(u)

    else:
        raise ValueError('unrecognized method')
    
def get_dirichlet_coeffs(u, dt, M, method='fft'):
    """
    Compute the Dirichlet coefficients of a trigonometric polynomial.

    Parameters
    ----------
    u : numpy.ndarray
        Input signal.
    dt : float
        Time resolution (s).
    M : int
        Trigonometric polynomial order.
    method : {'fft', 'inner'}
        Method to use when computing coefficients. The FFT method is
        generally faster than using inner products.

    Returns
    -------
    am : numpy.ndarray
        Array of `2*M+1` Dirichlet coefficients.
        
    Notes
    -----    
    Assumes that `u` is defined over times `dt*arange(0, len(u))` and
    that `dt*len(u)` is equal to the period of the trigonometric polynomial.

    """

    T = dt*len(u)

    if method == 'fft':
        u_fft = np.fft.fft(u)
        am = np.empty(2*M+1, np.complex)
        am[0:M] = u_fft[-M:]*dt/np.sqrt(T)
        am[M:] = u_fft[0:M+1]*dt/np.sqrt(T)
        return am
    
    elif method == 'inner':
        t = np.arange(0, len(u))*dt
        am = np.empty(2*M+1, np.complex)
        Omega = 2*np.pi*M/T
        for m in xrange(-M, M+1):
            am[m+M] = dt*np.sum(u*em(-m, t, Omega, M))
        return am

    else:
        raise ValueError('unrecognized method')

def gen_dirichlet_coeffs_2d(Mx, My):
    """
    Generate random Dirichlet coefficients for a 2d real signal.

    Parameters
    ----------
    Mx : int
        Trigonometric polynomial order along the X-axis.
    My : int
        Trigonometric polynomial order along the Y-axis.

    Returns
    -------
    c : numpy.ndarray
        Array of Dirichlet coefficients with shape `(2*My+1,
        2*Mx+1)`. The coefficients are ordered such that `c[0,0]`
        contains the coefficients for `mx == -Mx` and `my == -My`.
        
    """

    c = np.empty((2*My+1, 2*Mx+1), np.complex)

    c[0:My, 0:Mx] = crand(My, Mx)
    c[My+1:, Mx+1:] = np.rot90(np.rot90(np.conj(c[0:My, 0:Mx])))

    c[0:My, Mx+1:] = crand(My, Mx)
    c[My+1:, 0:Mx] = np.rot90(np.rot90(np.conj(c[0:My, Mx+1:])))

    c[My, Mx] = np.random.rand()

    c[My, 0:Mx] = crand(Mx)
    c[My, Mx+1:] = np.conj(c[My, 0:Mx][::-1])
    c[0:My, Mx] = crand(My)
    c[My+1:, Mx] = np.conj(c[0:My, Mx][::-1])

    return c

def gen_trig_poly_2d(Sx, Sy, dx, dy, c):
    """
    Construct a 2D trigonometric polynomial.

    Parameters
    ----------
    Sx : float
        Period of signal along the X-axis.
    Sy : float
        Period of signal along the Y-axis.
    dx : float
        Resolution along the X-axis.
    dy : float
        Resolution along the Y-axis.
    c : tuple or numpy.ndarray
        X-axis and Y-axis trigonometric polynomial orders or
        an array of Dirichlet coefficients with shape `(2*My+1, 2*Mx+1)`.

    Returns
    -------
    S : numpy.ndarray
        Generated signal.

    Notes
    -----
    This function uses the FFT to generate the output signal.

    See http://fourier.eng.hmc.edu/e101/lectures/Image_Processing/node6.html
    for an example of a DFT of a real 2D discrete signal.

    """

    if isinstance(c, tuple):
        if len(c) != 2:
            raise ValueError('invalid number of trigonometric polynomial orders')
        Mx, My = c
        if Mx < 1 or My < 1:
            raise ValueError('Mx and My must exceed 0')
        c = gen_dirichlet_coeffs_2d(Mx, My)
    elif np.iterable(c):
        if len(np.shape(c)) != 2:
            raise ValueError('coefficient array must have 2 dimensions')
        if np.shape(c)[0] % 2 == 0 or np.shape(c)[1] % 2 == 0:
            raise ValueError('coefficient array must have odd number of rows and columns')
        My = np.shape(c)[0]/2
        Mx = np.shape(c)[1]/2
    else:
        raise ValueError('unrecognized parameter type')
    
    Nx = int(np.ceil(Sx/dx))
    Ny = int(np.ceil(Sy/dy))
    S_fft = np.zeros((Ny, Nx), np.complex)

    S_fft[0:My+1, 0:Mx+1] = c[My:, Mx:]
    S_fft[-My:, -Mx:] = c[0:My, 0:Mx]
    S_fft[0:My+1, -Mx:] = c[-My-1:, 0:Mx]
    S_fft[-My:, 0:Mx+1] = c[0:My, -Mx-1:]
    
    return np.real(np.fft.ifft2(S_fft))

def get_dirichlet_coeffs_2d(S, Mx, My):
    """
    Compute the Dirichlet coefficients of a 2d trigonometric polynomial.

    Parameters
    ----------
    S : numpy.ndarray
        Input signal with shape `(Ny, Nx)`.
    Mx : int
        Trigonometric polynomial order along the X-axis.
    My : int
        Trigonometric polynomial order along the Y-axis.

    Returns
    -------
    c : numpy.ndarray
        Array of Dirichlet coefficients with shape `(2*My+1,
        2*Mx+1)`. The coefficients are ordered such that `c[0,0]`
        contains the coefficients for `mx == -Mx` and `my == -My`.
        
    Notes
    -----    
    Assumes that `S` is defined over the X-axis range `dx*arange(0,
    S.shape[1])` and Y-axis range `dy*arange(0, S.shape[0])` and
    that `dx*S.shape[1]` and `dy*S.shape[0]` are equal to the X-axis
    and Y-axis periods of the trigonometric polynomial, respectively.

    """

    S_fft = np.fft.fft2(S)

    c = np.empty((2*My+1, 2*Mx+1), np.complex)

    c[My:, Mx:] = S_fft[0:My+1, 0:Mx+1] 
    c[0:My, 0:Mx] = S_fft[-My:, -Mx:] 
    c[-My-1:, 0:Mx] = S_fft[0:My+1, -Mx:] 
    c[0:My, -Mx-1:] = S_fft[-My:, 0:Mx+1] 

    return c
