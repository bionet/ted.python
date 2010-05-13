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

__all__ = ['animate', 'animate2', 'animate_compare', 'frame_compare']

import numpy as np
import pylab as p

def animate(data, step=1):
    """Animate 2D video data in the N x N x M data array. Each
    N x N data[step, :, :] is treated as a 2D bitmap."""

    # Get maximum value in data to scale the luminance appropriately:
    mx = np.max(np.abs(data))
    img = p.imshow(data[0, :, :], vmin=-mx, vmax=mx)
    for k in np.arange(0, data.shape[0], step):
        img.set_data(data[k, :, :])
        p.draw()

def animate2(data_1, data_2, step=1):
    """Animate two 2D video data arrays simultaneously. The N x N
    arrays data_1[step, :, :] and data_2[step, :, :] are treated as 2D
    bitmaps."""

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
        img_1.set_data(data_1[k, :, :])
        img_2.set_data(data_2[k, :, :])
        p.draw()

def animate_compare(data_1, data_2, step=1):
    """Animate two 2D video data arrays simultaneously along with
    their difference. The N x N arrays data_1[step, :, :] and
    data_2[step, :, :] are treated as 2D bitmaps."""
    
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
        img_1.set_data(data_1[k, :, :])
        img_2.set_data(data_2[k, :, :])
        img_err.set_data(data_1[k, :, :]-data_2[k, :, :])
        p.draw()
    
def frame_compare(data_1, data_2, i=1):
    """Display frame i from video data arrays data_1 and data_2
    simultaneously."""

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
