#!/usr/bin/env python

"""
Demo of how to filter trigonometric polynomial signals.
"""

import numpy as np
import trig_poly as tp

# Set matplotlib backend so that plots can be generated without a
# display:
import matplotlib
matplotlib.use('AGG')

import bionet.utils.plotting as pl
output_name = 'filter_demo_'
output_count = 0
output_ext = '.png'

def make_gammatone(t, N, i):
    """
    Construct a gammatone filter.

    Constructs the impulse response of a gammatone filter
    from a filterbank of the specified size.
    
    Parameters
    ----------
    t : ndarray
        Times over which to compute the impulse responses.
    N : int
        Number of filters in filterbank.
    i : int
        Index of filter to compute.
        
    Returns
    -------
    h : ndarray
        Impulse response of the generated filter.

    """
    
    if i < 0 or i >= N:
        raise ValueError('invalid index i')

    Q_ear = 9.26449
    BW_min = 24.7
    beta = 1.019
    n = 4
    f_min = 200
    f_max = 1400

    ERB = lambda f_c: ((f_c/Q_ear)**n+(BW_min**n))**(1.0/n)

    o = (Q_ear/N)*(np.log(f_max+Q_ear*BW_min)-np.log(f_min+Q_ear*BW_min))
    f_c = -Q_ear*BW_min+(f_max+Q_ear*BW_min)*np.exp(-(o/Q_ear)*(N-1-i))
    h = t**(n-1)*np.exp(-2*np.pi*beta*ERB(f_c)*t)*np.cos(2*np.pi*f_c*t)
    return h/np.max(np.abs(np.fft.fft(h)))

def make_gammatone_fb(t, N):
    """
    Construct a gammatone filterbank.

    Constructs the impulse responses of all of the filters in a
    gammatone filterbank.

    t : ndarray
        Times over which to compute the impulse responses.
    N : int
        Number of filters in filterbank.
        
    Returns
    -------
    h : ndarray
        Impulse response of the generated filter.
    
    """
    h = np.zeros((N, len(t)))
    for i in xrange(N):
        h[i, :] = make_gammatone(t, N, i)
    return h

def filter_trig_poly(u, h, dt, M):
    """
    Filter a trigonometric polynomial with a filter's impulse response.

    Parameters
    ----------
    u : numpy.ndarray
        Input signal.
    h : numpy.ndarray
        Impulse response of filter.
    dt : float
        Time resolution.
    M : int
        Trigonometric polynomial order.
        
    Returns
    -------
    v : numpy.ndarray
        Filtered signal.
        
    Notes
    -----
    Assumes that `u` is defined over times `dt*arange(0, len(t))` and
    that `dt*len(u)` is equal to the period of the trigonometric polynomial.    

    """

    # Get the Dirichlet coefficients of the signal:
    am = get_dirichlet_coeffs_fft(u, dt, M)

    # Get the Dirichlet coefficients of the filter:
    hm = get_dirichlet_coeffs_fft(h, dt, M)

    # Construct the filtered signal using the above coefficients:
    T = dt*len(u)
    Omega = 2*np.pi*M/T
    t = np.arange(0, len(u))*dt
    v = np.zeros(len(u), np.complex)
    for m in xrange(-M, M+1):
        v += hm[m+M]*am[m+M]*em(m, t, Omega, M)*np.sqrt(T)/dt

    return np.real(v)

def filter_trig_poly_fft(u, h):
    """
    Filter a trigonometric signal with a filter's impulse response.

    Parameters
    ----------
    u : numpy.ndarray
        Input signal.
    h : numpy.ndarray
        Impulse response of filter.

    Returns
    -------
    v : numpy.ndarray
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

if __name__ == '__main__':
    
    print 'generating trigonometric polynomial signal..'
    M = 250
    Omega = 2*np.pi*2000
    T = 2*np.pi*M/Omega

    dt = 1e-5
    t = np.arange(0, T, dt)

    am = tp.gen_dirichlet_coeffs(M)
    u = tp.gen_trig_poly(t, am)

    # Try to recover the Dirichlet coefficients of the generated signal
    # using different methods. Note that this only works if u contains an
    # entire period of the signal (i.e., arange(0, T, dt)):
    print 'reconstructing signal from recovered coefficients..'
    am_fft = tp.get_dirichlet_coeffs_fft(u, dt, M)
    u_rec = tp.gen_trig_poly(t, am_fft)
    pl.plot_compare(t, u, u_rec, 'Signal Reconstruction Error',
                    output_name + str(output_count) + output_ext)
    output_count += 1 

    # Create a filter:
    h = make_gammatone(t, 16, 0)
    hm = tp.get_dirichlet_coeffs_fft(h, dt, M)
    h_rec = tp.gen_trig_poly(t, hm)
    pl.plot_compare(t, h, h_rec, 'Filter Reconstruction Error',
                    output_name + str(output_count) + output_ext)
    output_count += 1 
      
    # Filter the signal using FFTs:
    v_fft = filter_trig_poly_fft(u, h)
    pl.plot_signal(t, v_fft, 'Filtered Signal',
                   output_name + str(output_count) + output_ext)
