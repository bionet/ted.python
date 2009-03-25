#!/usr/bin/env python

"""
Test scipy extras.
"""

from numpy.testing import *
import numpy as np

import bionet.utils.scipy_extras as sce

class TestScipyExtras(TestCase):
    def test_ei(self):

        # XXX: Create test cases using data from Tables of the
        # Exponential Integral for Complex Arguments (Abramowitz & Stegun)
        assert_almost_equal(sce.ei(1), 1.8951178163559367555)
        assert_almost_equal(sce.ei(1j), 0.3374039229009681347 + 2.5168793971620796342j)
