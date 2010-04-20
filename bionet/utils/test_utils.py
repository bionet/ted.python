#!/usr/bin/env python

"""
Testing Utilities
=================

This module contains functions for testing the performance and results
of functions that produce numerical data.

- func_timer      Function execution timer. Can be used as a decorator.
- plot_encoded    Plot a time-encoded signal.
- plot_compare    Display two superimposed signals and their difference.
- plot_fourier    Plot the Discrete Time Fourier transform of a signal.
- plot_raster     Display several time sequences as a raster plot.
- plot_signal     Plot a signal over some time interval.

"""

__all__ = ['func_timer', 'plot_compare', 'plot_encoded',
           'plot_fourier','plot_raster', 'plot_signal']

import time
    
import numpy as np
import pylab as p

# Since the fft function in scipy is faster than that in numpy, try to
# import the former before falling back to the latter:
try:
    from scipy.fftpack import fft
except ImportError:
    from numpy.fft import fft

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
    t : ndarray of floats
        Times (s) at which the signal is defined.
    u : ndarray of floats
        Signal samples.

    Optional Parameters
    -------------------
    fig_title : string
        Plot title.
    file_name : string
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
    """Plot a time-encoded signal.

    Parameters
    ----------
    t : ndarray of floats
        Times (s) at which the original signal was sampled.
    u : ndarray of floats
        Signal samples.
    s : ndarray of floats
        Intervals between encoded signal spikes.

    Optional Parameters
    -------------------
    fig_title : string
        Plot title.
    file_name : string
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
    t : ndarray of floats
        Times (s) at which the signal is defined.
    u, v : ndarrays of floats
        Signal samples.

    Optional Parameters
    -------------------
    fig_title : string
        Plot title.
    file_name : string
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
    p.xlabel('t (s)')
    p.ylabel('error (dB)')
    p.gca().set_xlim(min(t), max(t))
    p.draw_if_interactive()
    if file_name:
        p.savefig(file_name)

def plot_fourier(u, fs, *args):
    """Plot the Discrete Fourier Transform of a signal.

    Parameters
    ----------
    u : ndarray of floats
        Sampled signal.
    fs : float
        Sampling rate (Hz).

    Optional Parameters
    -------------------
    fmin : float
        Minimum frequency to display (Hz).
    fmax : float:
        Maximum frequency to display (Hz).

    Notes
    -----
    This function may take a long time to run if the frequency range
    is very large.
    
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
    f = (fs/2)*np.arange(0, n)/n

    a = int(2.0*n*fmin/fs)
    b = int(2.0*n*fmax/fs)

    p.clf()
    p.subplot(211)
    p.stem(f[a:b], np.real(uf)[a:b])
    p.ylabel('real')
    p.subplot(212)
    p.stem(f[a:b], np.imag(uf)[a:b])
    p.ylabel('imag')
    p.xlabel('f (Hz)')

def plot_raster(ts_list, plot_stems=True, fig_title='', file_name=''):
    """Plot several time sequences as a raster.

    Parameters
    ----------
    ts_list : list of ndarrays
        Time sequences to plot.
    plot_stems : bool
        Show stems for all events.

    Optional Parameters
    -------------------
    fig_title : string
        Plot title.
    file_name : string
        File in which to save the plot.

    """

    M = len(ts_list)
    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
    max_ts = max([max(ts) for ts in ts_list])
    ax = p.gca()
    ax.axis([0, max_ts, -0.5, M-0.5])
    p.yticks(xrange(M))
    for (ts,y) in zip(ts_list,xrange(M)):
        p.axhline(y, 0, 1, color='b', hold=True)
        p.plot(ts, y*np.ones(len(ts)), '|b', hold=True,
               markersize=20,
               scalex=False, scaley=False)
        if plot_stems:
            for t in ts:
                ax.add_line(mp.lines.Line2D([t,t], [-0.5, y], color='r', linestyle=':'))
    ax.xaxis.set_major_locator(mp.ticker.MultipleLocator(10.0**np.ceil(np.log10(max_ts/10))))
    p.xlabel('t (s)')
    p.title(fig_title)
    p.draw_if_interactive()
    if file_name:
        p.savefig(file_name)
