#!/usr/bin/env python

import os
import sys
import time

import numpy as np

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages, Extension
from distutils.command.install import INSTALL_SCHEMES
from distutils.sysconfig import get_python_version
from Cython.Distutils import build_ext

# This enables the installation of __init__.py files in
# namespace packages:
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['platlib']

NAME =               'bionet.ted'
VERSION =            '0.07'
AUTHOR =             'Lev Givon'
AUTHOR_EMAIL =       'lev@columbia.edu'
URL =                'http://bionet.github.com/'
MAINTAINER =         'Lev Givon'
MAINTAINER_EMAIL =   'lev@columbia.edu'
DESCRIPTION =        'Time Encoding and Decoding Toolbox'
DOWNLOAD_URL =       URL
LICENSE =            'BSD'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Topic :: Scientific/Engineering',
    'Topic :: Software Development']

if sys.platform in ['linux2', 'darwin']:
    ext_name = 'bionet.ted.bpa_cython_' + sys.platform

    # Need numpy include files to compile BPA extension:
    bpa_cython = Extension(ext_name,
                           ['bionet/ted/bpa_cython.pyx'],
                           [np.get_include()],
                           libraries=['python' + get_python_version()])
else:
    bpa_cython = None

if __name__ == '__main__':
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name = NAME,
          version = VERSION,
          author = AUTHOR,
          author_email = AUTHOR_EMAIL,
          url = URL,
          maintainer = MAINTAINER,
          maintainer_email = MAINTAINER_EMAIL,
          description = DESCRIPTION,
          license = LICENSE,
          classifiers = CLASSIFIERS,
          packages = find_packages(),
          data_files = [('bionet', ['bionet/__init__.py'])],
          namespace_packages = ['bionet'],
          install_requires = ['numpy >= 1.2.0', 
                              'scipy >= 0.7.0'],
          extras_require = dict(
              matplotlib = 'matplotlib >= 0.98',
              opencv = 'opencv >= 2.1.0',
              tables = 'tables >= 2.1.1'),
          ext_modules = [bpa_cython],
          cmdclass = {'build_ext': build_ext})
