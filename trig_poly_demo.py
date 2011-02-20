#!/usr/bin/env python

"""
Trigonometric polynomial demo.
"""

import numpy as np
import trig_poly as tp

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
                 
# Generate a trigonometric polynomial:
M = 250
Omega = 2*np.pi*2000
TM = 2*np.pi*M/Omega

dt = 1e-5
t = np.arange(0, TM, dt)

am = tp.gen_trig_poly_coeffs(M)
u = tp.gen_trig_poly(t, Omega, am)

# Try to recover the Dirichlet coefficients of the generated signal
# using different methods. Note that this only works if u contains an
# entire period of the signal (i.e., arange(0, TM, dt)):
am_inner = tp.get_dir_coeff_inner(u, dt, Omega, M)
am_fft = tp.get_dir_coeff_fft(u, dt, Omega, M)

print 'Successfully recovered Dirichlet coefficients ' + \
      'using inner products: ', np.allclose(am, am_inner)
print 'Successfully recovered Dirichlet coefficients ' + \
      'using FFTs: ', np.allclose(am, am_fft)

# Create a filter:
h = make_gammatone(t, 16, 0)
hm = tp.get_dir_coeff_fft(h, dt, Omega, M)
h_rec = tp.gen_trig_poly(t, Omega, hm)

print 'Successfully constructed filter from Dirichlet coefficients: ', \
      np.allclose(h, h_rec)
      
# Compare the results of filtering with Dirichlet coefficients to
# scipy's filtering function:
v = tp.filter_trig_poly(u, h, dt, Omega, M)
v2 = tp.filter_trig_poly_fft(u, h)

