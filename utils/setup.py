#!/usr/bin/env python

import os

# Install setuptools if it isn't available:
try:
    import setuptools
except ImportError:
    from distribute_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages
from distutils.command.install import INSTALL_SCHEMES

# This enables the installation of __init__.py files in
# namespace packages:
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

NAME =               'bionet.utils'
VERSION =            '0.017'
AUTHOR =             'Lev Givon'
AUTHOR_EMAIL =       'lev@columbia.edu'
URL =                'http://bionet.github.com/'
MAINTAINER =         'Lev Givon'
MAINTAINER_EMAIL =   'lev@columbia.edu'
DESCRIPTION =        'Bionet utilities'
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
      install_requires = ['matplotlib >= 0.98',
                          'numpy >= 1.2.0', 
                          'scipy >= 0.7.0',
                          'tables >= 2.1.1'])
