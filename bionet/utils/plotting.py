#!/usr/bin/env python

"""
Plotting Utilities
==================
This module contains functions for plotting the results of functions
that produce numerical data.

- plot_compare    Display two superimposed signals and their difference.
- plot_encoded    Plot a time-encoded signal.
- plot_fourier    Plot the Discrete Time Fourier transform of a signal.
- plot_raster     Display several time sequences as a raster plot.
- plot_signal     Plot a signal over some time interval.

The module also contains several wrappers for matplotlib's 3D plotting
facilities.

- contour         Create a 3D contour plot.
- contourf        Create a filled 3D contour plot.
- surf            Create a 3D surface plot.
- wireframe       Create a 3D wireframe plot.
"""

# Copyright (c) 2009-2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

__all__ = ['plot_compare', 'plot_encoded',
           'plot_fourier','plot_raster', 'plot_signal',
           'contour', 'contourf', 'surf', 'wireframe']

import numpy as np
import pylab as p

from mpl_toolkits.mplot3d import axes3d
import matplotlib as mp

# Since the fft function in scipy is faster than that in numpy, try to
# import the former before falling back to the latter:
try:
    from scipy.fftpack import fft
except ImportError:
    from numpy.fft import fft

def plot_signal(t, u, fig_title='', file_name=''):
    """
    Plot a signal.

    Parameters
    ----------
    t : ndarray of floats
        Times (in s) at which the signal is defined.
    u : ndarray of floats
        Signal samples.
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
    """
    Plot a time-encoded signal.

    Parameters
    ----------
    t : ndarray of floats
        Times (in s) at which the original signal was sampled.
    u : ndarray of floats
        Signal samples.
    s : ndarray of floats
        Intervals between encoded signal spikes.
    fig_title : string
        Plot title.
    file_name : string
        File in which to save the plot.

    Notes
    -----
    The spike times (i.e., the cumulative sum of the interspike
    intervals) must all occur within the interval `t-min(t)`.

    """

    dt = t[1]-t[0]
    cs = np.cumsum(s)
    if cs[-1] >= max(t)-min(t):
        raise ValueError('some spike times occur outside of signal''s support')

    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
    p.axes([0.125, 0.3, 0.775, 0.6])
    p.vlines(cs+min(t), np.zeros(len(cs)), u[np.asarray(cs/dt, int)], 'b')
    p.hlines(0, 0, max(t), 'r')
    p.plot(t, u, hold=True)
    p.xlabel('t (s)')
    p.ylabel('u(t)')
    p.title(fig_title)
    p.gca().set_xlim(min(t), max(t))
    a = p.axes([0.125, 0.1, 0.775, 0.1])
    p.plot(cs+min(t), np.zeros(len(s)), 'ro')
    a.set_yticklabels([])
    p.xlabel('%d spikes' % len(s))
    p.gca().set_xlim(min(t), max(t))
    p.draw_if_interactive()
    if file_name:
        p.savefig(file_name)

def plot_compare(t, u, v, fig_title='', file_name=''):
    """
    Compare two signals and plot the difference between them.

    Parameters
    ----------
    t : ndarray of floats
        Times (s) at which the signal is defined.
    u, v : ndarrays of floats
        Signal samples.
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

def plot_fourier(u, fs, fmin=0.0, fmax=None, style='line'):
    """
    Plot the Discrete Fourier Transform of a signal.

    Parameters
    ----------
    u : ndarray of floats
        Sampled signal.
    fs : float
        Sampling rate (Hz).
    fmin : float
        Minimum frequency to display (Hz).
    fmax : float:
        Maximum frequency to display (Hz).
    style : {'line', 'semilogy', 'stem'}
        Set plot style.
        
    Notes
    -----
    This function may take a long time to run if the frequency range
    is very large.

    """

    if fmin < 0.0 or fmin >= fs/2:
        raise ValueError('invalid minimum frequency')

    if fmax is None:
        fmax = fs/2
    if fmax <= fmin or fmax > fs/2:
        raise ValueError('invalid maximum frequency')

    n = len(u)/2
    uf = fft(u)[0:n]
    f = (fs/2.0)*np.arange(0, n)/n

    a = int(2.0*n*fmin/fs)
    b = int(2.0*n*fmax/fs)

    p.clf()
    p.subplot(211)
    if style == 'stem':
        p.stem(f[a:b], np.real(uf)[a:b])
        p.ylabel('real')
    elif style == 'semilogy':
        p.semilogy(f[a:b], np.abs(np.real(uf)[a:b]))
        p.ylabel('|real|')
    else:
        p.plot(f[a:b], np.real(uf)[a:b])
        p.ylabel('real')
    p.xlim((f[a], f[b-1]))
    p.subplot(212)
    if style == 'stem':
        p.stem(f[a:b], np.imag(uf)[a:b])
        p.ylabel('imag')
    elif style == 'semilogy':
        p.semilogy(f[a:b], np.abs(np.imag(uf)[a:b]))
        p.ylabel('|imag|')
    else:
        p.plot(f[a:b], np.imag(uf)[a:b])
        p.ylabel('imag')
    p.xlim((f[a], f[b-1]))
    p.xlabel('f (Hz)')

def plot_raster(ts_list, plot_stems=True, plot_axes=True, marker='.', markersize=5, fig_title='', file_name=''):
    """
    Plot several time sequences as a raster.

    Parameters
    ----------
    ts_list : list of ndarrays
        Time sequences to plot.
    plot_stems : bool
        Show stems for all events.
    plot_axes : bool
        Show horizontal axes for all sequences.
    marker : char
        Marker symbol.
    markersize : int
        Marker symbol size.
    fig_title : string
        Plot title.
    file_name : string
        File in which to save the plot.

    """

    M = len(ts_list)
    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
    max_ts = max([max(ts) if len(ts) > 1 else 0 for ts in ts_list])
    ax = p.gca()
    ax.axis([0, max_ts, -0.5, M-0.5])
    p.yticks(xrange(M))
    for (ts,y) in zip(ts_list,xrange(M)):
        if plot_axes:
            p.axhline(y, 0, 1, color='b', hold=True)
        p.plot(ts, y*np.ones(len(ts)), marker+'b', hold=True,
               markersize=markersize,
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

def contour(X, Y, Z, *args):

    # For some reason, this is necessary to prevent clf() from raising
    # an exception:    
    try:
        p.clf()
    except ValueError:
        p.clf()
    ax = p.gca(projection='3d')
    ax.contour(X, Y, Z, *args)
    p.draw_if_interactive()
contour.__doc__ = axes3d.Axes3D.contour.__doc__

def contourf(X, Y, Z, *args):

    # For some reason, this is necessary to prevent clf() from raising
    # an exception:
    try:
        p.clf()
    except ValueError:
        p.clf()
    ax = p.gca(projection='3d')
    ax.contourf(X, Y, Z, *args)
    p.draw_if_interactive()
contourf.__doc__ = axes3d.Axes3D.contourf.__doc__

def surf(X, Y, Z, *args):

    # For some reason, this is necessary to prevent clf() from raising
    # an exception:    
    try:
        p.clf()
    except ValueError:
        p.clf()
    ax = p.gca(projection='3d')
    ax.plot_surface(X, Y, Z, *args)
    p.draw_if_interactive()
surf.__doc__ = axes3d.Axes3D.plot_surface.__doc__

def wireframe(X, Y, Z, *args):

    # For some reason, this is necessary to prevent clf() from raising
    # an exception:
    try:
        p.clf()
    except ValueError:
        p.clf()
    ax = p.gca(projection='3d')
    ax.plot_wireframe(X, Y, Z, *args)
    p.draw_if_interactive()
wireframe.__doc__ = axes3d.Axes3D.plot_wireframe.__doc__
