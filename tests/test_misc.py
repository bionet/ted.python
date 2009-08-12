#!/usr/bin/env python

"""
Test classes for miscellaneous functions and classes.
"""

import unittest
import bionet.utils.misc as m

class TestMisc(unittest.TestCase):
    def testChunks(self):
        x = range(10)
        i = m.chunks(x,5)
        x1 = i.next()
        x2 = i.next()
        assert(x1 == range(5) and x2 == range(5,10))

    def testSerialBufferInit(self):
        x = range(100)
        i = iter(x)
        sb = m.SerialBuffer(i.next, 50)
        assert(len(sb) == 50)

    def testSerialBufferRead(self):
        x = range(10)
        i = iter(x)
        sb = m.SerialBuffer(i.next)
        x1 = sb.read(5)
        x2 = sb.read(5)
        x3 = sb.read(5)
        assert(x1 == range(5) and x2 == range(5,10) and x3 == [])

if __name__ == "__main__":
    unittest.main()
