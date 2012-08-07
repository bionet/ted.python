#!/usr/bin/env python

"""
Test numpy extras.
"""

import numpy as np
from numpy.testing import *
from unittest import main

import bionet.utils.numpy_extras as npe

class TestNumpyExtras(TestCase):
    def test_mdot(self):
        x = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        assert_equal(npe.mdot(x, x, x),
                     np.array([[468, 576, 684],
                               [1062, 1305, 1548],
                               [1656, 2034, 2412]]))

    def test_rank(self):
        x = np.array([[1, 1, 1, 1], [1, 2, 3, 4],
                      [2, 2, 2, 2], [3, 4, 5, 6]], dtype=np.float)
        assert_equal(npe.rank(x), 2)
        x = np.array([[1, 2, 3, 4], [7, 3, 4, 5],
                      [3, 4, 5, 6], [4, 5, 0, 7]], dtype=np.float)
        assert_equal(npe.rank(x), 4)

    def test_mpower(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_almost_equal(npe.mpower(x, 2),
                            np.array([[30.0, 36.0, 42.0],
                                      [66.0, 81.0, 96.0],
                                      [102.0, 126.0, 150.0]]))
        assert_almost_equal(npe.mpower(x, 2.5),
                            np.array([[116.825+0.950j, 143.544+0.257j,
                                       170.264-0.434j],
                                      [264.563+0.104j, 325.072+0.028j,
                                       385.581-0.048j],
                                      [412.301-0.740j, 506.600-0.200j,
                                       600.899+0.338j]]), 3)

if __name__ == "__main__":
    main()
