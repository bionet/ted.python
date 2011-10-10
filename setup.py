#!/usr/bin/env python

import os
import sys
import time

import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc, get_python_lib
from Cython.Distutils import build_ext

NAME =               'bionet.ted'
VERSION =            '0.062'
IS_RELEASED =        True
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

def get_version(version, is_released=True):
    """
    Return package version. If `is_released` is True, append the
    package Mercurial revision.
    """

    full_version = version
    if not is_released and os.path.exists('.hg'):

        # Do not read any config file when running mercurial:
        os.environ['HGRCPATH'] = '' 
        cmd = '%s /usr/bin/hg id -it' % sys.executable

        try:
            l = os.popen(cmd).read().split()
        except OSError, e:
            print "warning: could not determine version: %s" % e

        # Toss non-numbered tags:
        while len(l) > 1 and l[-1][0].isalpha(): 
            l.pop()
        if l:

            # Get the latest tag or revision number:
            version = l[-1] 

            # Append the current time if the source code has been
            # modified but not checked in:
            if version.endswith('+'):
                version += time.strftime('%Y%m%d')

        # Append the global Mercurial revision:
        full_version += '.dev.'+version

    return full_version

def write_version_py(version, filename='bionet/ted/version.py'):
    """
    Write the indicated package version to the specified Python file.
    """
    
    f = file(filename,"w")
    f.write("# THIS FILE IS GENERATED BY THE TED SETUP.PY FILE\n")
    f.write("__version__ = '" + version + "'\n")
    f.close()
    
if sys.platform in ['linux2', 'darwin']:
    ext_name = 'bionet.ted.bpa_cython_' + sys.platform

    # Need numpy include files to compile BPA extension:
    bpa_cython = Extension(ext_name,
                           ['bionet/ted/bpa_cython.pyx'],
                           [get_python_inc(), np.get_include()])
else:
    bpa_cython = None

if __name__ == '__main__':
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    write_version_py(get_version(VERSION, IS_RELEASED))
    
    setup(name = NAME,
          version = get_version(VERSION, IS_RELEASED),
          author = AUTHOR,
          author_email = AUTHOR_EMAIL,
          url = URL,
          maintainer = MAINTAINER,
          maintainer_email = MAINTAINER_EMAIL,
          description = DESCRIPTION,
          license = LICENSE,
          classifiers = CLASSIFIERS,

          packages = ['bionet', 'bionet.ted'],
          package_dir = {'bionet.ted': 'bionet/ted'},
          ext_modules = [bpa_cython],
          cmdclass = {'build_ext': build_ext})
