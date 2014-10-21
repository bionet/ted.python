#!/usr/bin/env python

"""
Wrapper for BPA implementations. This module determines which implementation
to load depending on the platform; if the high-performance binary modules
cannot be loaded, a stock Python implementation is loaded instead.
"""

__all__ = ['bpa']

import sys

if 'linux' in sys.platform:
    try:
        from bionet.ted.bpa_cython_linux2 import *
    except ImportError:
        from bionet.ted.bpa_python import *
elif sys.platform == 'darwin':
    try:
        from bionet.ted.bpa_cython_darwin import *
    except ImportError:
        from bionet.ted.bpa_python import *
else:
    from bionet.ted.bpa_python import *

