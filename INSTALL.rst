.. -*- rst -*-

Installation Instructions
=========================

Obtaining the Latest Software
-----------------------------
The latest version of the Time Encoding and Decoding Toolkit can be
downloaded from the Bionet Group's `code repository page
<http://bionet.github.io/>`_.

Prerequisites
-------------

The Python implementation of the Time Encoding and Decoding Toolkit
requires that several software packages be present in order to be
built and installed (older versions of these packages may work, but have not
been tested):

* `Cython <http://www.cython.org>`_ 0.11.2 or later.
* `Numpy <http://numpy.scipy.org>`_ 1.2.0 or later.
* `Python <http://www.python.org>`_ 2.5 or later.
* `Scipy <http://www.scipy.org>`_ 0.7.0 or later.

To run the CUDA-dependent implementations, you will also need

* `NVIDIA CUDA Toolkit <http://www.nvidia.com/object/cuda_get.html>`_ 3.2 or later.
* `PyCUDA <http://mathema.tician.de/software/pycuda>`_ 0.94.2 or later.
* `scikit.cuda <http://www.bionet.ee.columbia.edu/code>`_ 0.04 or later.

To run the demo code and generate plots, the following package 
is also required:

* `Matplotlib <http://matplotlib.sourceforge.net>`_ 0.98 or later.

Some of the utility functions may require the following packages:

* `MEncoder <http://www.mplayerhq.hu/>`_ 1.0 or later.
* `PyTables <http://www.pytables.org>`_ 2.1.1 or later.
* `OpenCV <http://opencv.willowgarage.com/>`_ 2.1.0 or later.

To build the documentation, the following packages are also required:

* `Docutils <http://docutils.sourceforge.net>`_ 0.5 or later.
* `Jinja2 <htt://jinja.pocoo.org>`_ 2.2 or later
* `Pygments <http://pygments.org>`_ 0.8 or later
* `Sphinx <http://sphinx.pocoo.org>`_ 1.0.1 or later.
* `Sphinx Bootstrap Theme
  <https://github.com/ryan-roemer/sphinx-bootstrap-theme>`_ 0.2.6 or later.

This software has been tested on Linux; it should also work
on other platforms supported by the above packages.

Building and Installation
-------------------------

To build and install the toolkit, download and unpack the source release and 
run::

   python setup.py install

from within the main directory in the release. Sample code demonstrating how to 
use the toolkit is located in the
``demos/`` subdirectory. If you have `pip <http://www.pip-installer.org>`_
installed, you can install the latest package code directly from Github as
follows::

    pip install git+git://github.com/bionet/ted.python.git

To rebuild the documentation, run::

   make

from within the ``docs/`` subdirectory and follow the directions.

