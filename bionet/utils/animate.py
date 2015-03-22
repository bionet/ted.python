#!/usr/bin/env python

"""
Video Animation Functions
=========================
This module contains functions for displaying sequences of 2D data as
animations.

- animate           Animate a 2D video sequence.
- animate2          Animate two 2D video sequences simultaneously.
- animate_compare   Animate two 2D video sequences and their difference.
- frame_compare     Display frames from two sequences and their difference.
"""

# Copyright (c) 2009-2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

__all__ = ['animate', 'animate2', 'animate_compare', 'frame_compare']

import time
import numpy as np
import pylab as p

# Fix for animation problems with Qt4Agg backend:
if p.get_backend() == 'Qt4Agg':
    from PyQt4.QtGui import QApplication
    animate_fix = QApplication.processEvents
else:
    def animate_fix():
        pass

def animate(data, step=1, delay=0):
    """
    Animate sequence of frames.

    Animate a sequence of `Ny x Nx` bitmap frames stored in a `M x Ny x Nx` data
    array.

    Parameters
    ----------
    data : numpy.ndarray
        Sequence of `M` 2D bitmaps stored as an array with shape
        `(M, Ny, Nx)`.
    step : int
        Skip `step` frames between each displayed frames.
    delay : float
        Wait `delay` seconds between each frame refresh.
    """

    # Get maximum value in data to scale the luminance appropriately:
    mx = np.max(np.abs(data))
    img = p.imshow(data[0, :, :], vmin=-mx, vmax=mx)
    for k in np.arange(0, data.shape[0], step):
        time.sleep(delay)
        img.set_data(data[k, :, :])
        p.draw()
        animate_fix()

def animate2(data_1, data_2, step=1, delay=0):
    """
    Animate two sequence of frames simultaneously.

    Animate two sequences of `Ny x Nx` bitmap frames stored in two `M x Ny x Nx` data
    arrays.

    Parameters
    ----------
    data_1 : numpy.ndarray
        Sequence of `M` 2D bitmaps stored as an array with shape
        `(M, Ny, Nx)`.
    data_2 : numpy.ndarray
        Sequence of `M` 2D bitmaps stored as an array with shape
        `(M, Ny, Nx)`.
    step : int
        Skip `step` frames between each displayed frames.
    delay : float
        Wait `delay` seconds between each frame refresh.
    """

    if data_1.shape != data_2.shape:
        raise ValueError('cannot animate two video sequences with '
                         'different dimensions')

    # Get maximum value in data to scale the luminance appropriately:
    mx_1 = np.max(np.abs(data_1))
    mx_2 = np.max(np.abs(data_2))
    p.subplot(121)
    img_1 = p.imshow(data_1[0, :, :], vmin=-mx_1, vmax=mx_1)
    p.subplot(122)
    img_2 = p.imshow(data_2[0, :, :], vmin=-mx_2, vmax=mx_2)
    for k in np.arange(0, data_1.shape[0], step):
        time.sleep(delay)
        img_1.set_data(data_1[k, :, :])
        img_2.set_data(data_2[k, :, :])
        p.draw()
        animate_fix()

def animate_compare(data_1, data_2, step=1, delay=0):
    """
    Animate two sequence of frames and their difference simultaneously.

    Animate two sequences of `Ny x Nx` bitmap frames stored in two `M x Ny x Nx` data
    arrays simultaneously with their difference.

    Parameters
    ----------
    data_1 : numpy.ndarray
        Sequence of `M` 2D bitmaps stored as an array with shape
        `(M, Ny, Nx)`.
    data_2 : numpy.ndarray
        Sequence of `M` 2D bitmaps stored as an array with shape
        `(M, Ny, Nx)`.
    step : int
        Skip `step` frames between each displayed frames.
    delay : float
        Wait `delay` seconds between each frame refresh.
    """

    if data_1.shape != data_2.shape:
        raise ValueError('cannot animate two video sequences with '
                         'different dimensions')

    # Get maximum value in data to scale the luminance appropriately:
    mx_1 = np.max(np.abs(data_1))
    mx_2 = np.max(np.abs(data_2))
    mx_err = max(mx_1, mx_2)
    p.subplot(131)
    img_1 = p.imshow(data_1[0, :, :], vmin=-mx_1, vmax=mx_1)
    p.subplot(132)
    img_2 = p.imshow(data_2[0, :, :], vmin=-mx_2, vmax=mx_2)
    p.subplot(133)
    img_err = p.imshow(data_1[0, :, :]-data_2[0, :, :], vmin=-mx_err, vmax=mx_err)
    for k in np.arange(0, data_1.shape[0], step):
        time.sleep(delay)
        img_1.set_data(data_1[k, :, :])
        img_2.set_data(data_2[k, :, :])
        img_err.set_data(data_1[k, :, :]-data_2[k, :, :])
        p.draw()
        animate_fix()

def frame_compare(data_1, data_2, i=0):
    """
    Compare corresponding frames in two video sequences.

    Simultaneously display two corresponding frames from two video sequences of
    identical length.

    Parameters
    ----------
    data_1 : numpy.ndarray
        Sequence of `M` 2D bitmaps stored as an array with shape
        `(M, Ny, Nx)`.
    data_2 : numpy.ndarray
        Sequence of `M` 2D bitmaps stored as an array with shape
        `(M, Ny, Nx)`.
    i : int
        Index of frame to display.
    """

    if data_1.shape != data_2.shape:
        raise ValueError('cannot compare frames from two video sequences with '
                         'different dimensions')

    mx_1 = np.max(np.abs(data_1))
    mx_2 = np.max(np.abs(data_2))
    p.subplot(121)
    p.imshow(data_1[i, :, :], vmin=-mx_1, vmax=mx_1)
    p.subplot(122)
    p.imshow(data_2[i, :, :], vmin=-mx_2, vmax=mx_2)
    p.draw()
