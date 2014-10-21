#!/usr/bin/env python

import os
import sys
import time

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages, Extension
from distutils.command.install import INSTALL_SCHEMES
from distutils.sysconfig import get_python_version

# This is overwritten by Cython.Distutils.build_ext during package installation:
from distutils.command import build_ext

# This enables the installation of __init__.py files in
# namespace packages:
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['platlib']

NAME =               'bionet.ted'
VERSION =            '0.7.1'
AUTHOR =             'Lev Givon'
AUTHOR_EMAIL =       'lev@columbia.edu'
URL =                'https://github.com/bionet/ted.python/'
MAINTAINER =         'Lev Givon'
MAINTAINER_EMAIL =   'lev@columbia.edu'
DESCRIPTION =        'Time Encoding and Decoding Toolkit'
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

# Don't attempt to import numpy when it isn't actually needed; this enables pip
# to install numpy before bottleneck:
ext_modules = []
if not(len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or \
       sys.argv[1] in ('--help-commands', 'egg_info', '--version', 'clean'))):

    # Needed to build pyx files:
    from Cython.Distutils import build_ext

    if sys.platform in ['linux2', 'darwin']:

        # Need numpy include files to compile BPA extension:
        import numpy as np
        ext_name = 'bionet.ted.bpa_cython_' + sys.platform
        ext_modules = [Extension(ext_name,
                                 ['bionet/ted/bpa_cython.pyx'],
                                 [np.get_include()],
                                 libraries=['python' + get_python_version()])]

metadata = dict(name = NAME,
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
                install_requires = ['cython >= 0.20.0',
                                    'numpy >= 1.2.0',
                                    'scipy >= 0.7.0'],
                extras_require = dict(
                    matplotlib = 'matplotlib >= 0.98',
                    opencv = 'opencv >= 2.1.0',
                    tables = 'tables >= 2.1.1'),
                ext_modules = ext_modules,
                cmdclass = {'build_ext': build_ext})

if __name__ == '__main__':
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(**metadata)
