#!/usr/bin/env python

"""
Test scipy extras.
"""

import numpy as np
from numpy.testing import *
from unittest import main

import bionet.utils.scipy_extras as sce

class TestScipyExtras(TestCase):
    def test_ei(self):

        # XXX: Create test cases using data from Tables of the
        # Exponential Integral for Complex Arguments (Abramowitz & Stegun)
        assert_almost_equal(sce.ei(1), 1.8951178163559367555)
        assert_almost_equal(sce.ei(1j), 0.3374039229009681347 + 2.5168793971620796342j)

if __name__ == "__main__":
    main()

