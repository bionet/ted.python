#!/usr/bin/env python

"""
Various functions for testing performance and results.
"""

__all__ = ['func_timer', 'plot_signal', 'plot_encoded',
           'plot_compare', 'plot_fourier']

import time

import numpy as np
import pylab as p

def func_timer(f):
    """Time the execution of function f. If arguments are specified,
    they are passed to the function being timed."""

    def wrapper(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        stop = time.time()
        print 'execution time = %.3f s' % (stop-start)
        return res
    return wrapper

def plot_signal(t, u, fig_title='', file_name=''):
    """Plot a signal.

    Parameters
    ----------
    t: ndarray of floats
        Times (s) at which the signal is defined.
    u: ndarray of floats
        Signal samples.
    fig_title: string
        Plot title.
    file_name: string
        File in which to save the plot.

    """
    
    p.clf()

    # Set the plot window title:
    p.gcf().canvas.set_window_title(fig_title)
    p.plot(t, u)
    p.xlabel('t (s)')
    p.ylabel('u(t)')
    p.title(fig_title)

    # Make the x axis exactly correspond to the plotted signal's time range:
    p.gca().set_xlim(min(t), max(t)) 
    p.draw_if_interactive()
    if file_name:
        p.savefig(file_name)

def plot_encoded(t, u, s, fig_title='', file_name=''):
    """Plot a time encoded signal.

    Parameters
    ----------
    t: ndarray of floats
        Times (s) at which the original signal was sampled.
    u: ndarray of floats
        Signal samples.
    s: ndarray of floats
        Intervals between encoded signal spikes.
    fig_title: string
        Plot title.
    file_name: string
        File in which to save the plot.

    """

    dt = t[1]-t[0]
    cs = np.cumsum(s)
    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
    p.axes([0.125, 0.3, 0.775, 0.6])
    p.vlines(cs, np.zeros(len(cs)), u[np.asarray(cs/dt, int)], 'b')
    p.hlines(0, 0, max(t), 'r')
    p.plot(t, u, hold=True)
    p.xlabel('t (s)')
    p.ylabel('u(t)')
    p.title(fig_title)
    p.gca().set_xlim(min(t), max(t))
    a = p.axes([0.125, 0.1, 0.775, 0.1])
    p.plot(cs, np.zeros(len(s)), 'ro')
    a.set_yticklabels([])
    p.xlabel('%d spikes' % len(s))
    p.gca().set_xlim(min(t), max(t))
    p.draw_if_interactive()
    if file_name:
        p.savefig(file_name)

def plot_compare(t, u, v, fig_title='', file_name=''):
    """Compare two signals and plot the difference between them.

    Parameters
    ----------
    t: ndarray of floats
        Times (s) at which the signal is defined.
    u,v: ndarrays of floats
        Signal samples.
    fig_title: string
        Plot title.
    file_name: string
        File in which to save the plot.

    """
    
    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
    p.subplot(211)
    p.plot(t, u, 'b', t, v, 'r')
    p.xlabel('t (s)')
    p.ylabel('u(t)')
    p.title(fig_title)
    p.gca().set_xlim(min(t), max(t))
    p.subplot(212)
    p.plot(t, 20*np.log10(abs(u-v)))
    p.xlabel('t')
    p.ylabel('error (dB)')
    p.gca().set_xlim(min(t), max(t))
    p.draw_if_interactive()
    if file_name:
        p.savefig(file_name)

def plot_fourier(u, fs, *args):
    """Plot the Discrete Fourier Transform of a signal.

    Parameters
    ----------
    u: ndarray of floats
        Sampled signal.
    fs: float
        Sampling rate (Hz).

    Optional Parameters
    -------------------
    fmin: float
        Minimum frequency to display (Hz).
    fmax: float:
        Maximum frequency to display (Hz).

    """

    if len(args) > 0:
        fmin = args[0]
    else:
        fmin = 0.0
    if fmin < 0.0 or fmin >= fs/2:
        raise ValueError('invalid minimum frequency')

    if len(args) == 2:
        fmax = args[1]
    else:
        fmax = fs/2
    if fmax <= fmin or fmax > fs/2:
        raise ValueError('invalid maximum frequency')
    
    n = len(u)/2
    uf = fft(u)[0:n]
    f = (fs/2)*arange(0, n)/n

    a = int(2.0*n*fmin/fs)
    b = int(2.0*n*fmax/fs)

    p.clf()
    p.subplot(211)
    p.stem(f[a:b], real(uf)[a:b])
    p.ylabel('real')
    p.subplot(212)
    p.stem(f[a:b], imag(uf)[a:b])
    p.ylabel('imag')
    p.xlabel('f (Hz)')

