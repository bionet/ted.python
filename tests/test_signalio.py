#!/usr/bin/env python

"""
Test classes for writing and reading signals to and from HDF files.
"""

import unittest
import os

import numpy as np
import utils.signalio as s

filename = 'test_signalio_data.h5'
blocksize = 10000

class SignalIOTestCase(unittest.TestCase):
    def setUp(self):
        '''Generate and save test data.'''

        N = 1000000
        self.u = np.random.rand(N)

        w = s.WriteArray(filename)
        w.write(self.u)
        w.close()

    def tearDown(self):
        '''Clean up test file.'''

        os.remove(filename)

    def testReadOneBlock(self):
        '''Test one-block read of saved data.'''
        r = s.ReadArray(filename)
        u_read = r.read()
        r.close()

        assert all(self.u==u_read),'read block does not match original block'

    def testReadManyBlocks(self):
        '''Test multi-block read of saved data.'''
        
        r = s.ReadArray(filename,blocksize)
        temp = []
        while True:
            datablock = r.read()
            if not len(datablock):
                break
            temp += datablock.tolist()
        u_read = np.array(temp)
        r.close()

        assert all(self.u==u_read),'read block does not match original block'

if __name__ == "__main__":
    unittest.main()
