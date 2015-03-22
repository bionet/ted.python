#!/usr/bin/env python

"""
Miscellaneous Functions
=======================
This module contains various unclassified utility functions and classes.

- chunks           Return a generator that splits a sequence into chunks.
- func_timer       Function execution timer. Can be used as a decorator.
- SerialBuffer     Buffer interface to a serial data source.
"""

# Copyright (c) 2009-2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

__all__ = ['chunks', 'func_timer', 'SerialBuffer']

import time

def chunks(seq, n):
    """
    Chunk generator.

    Return a generator whose `next()` method returns length `n`
    subsequences of the given sequence.  If `len(seq) % n != 0`, the
    last subsequence returned will contain fewer than `n` entries.

    Parameters
    ----------
    seq : iterable
        Sequence to split into chunks.
    n : int
        Chunk size.

    Returns
    -------
    g : generator
        Generator that will return length `n` chunks of `seq`.
    """

    for i in xrange(0, len(seq), n):
        yield seq[i:i+n]

def func_timer(f):
    """
    Time the execution of a function.

    Parameters
    ----------
    f : function
        Function to time.
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        stop = time.time()
        print 'execution time = %.5f s' % (stop-start)
        return res
    return wrapper

class SerialBuffer:
    """
    Serial buffer class.

    This class implements a buffer that automatically replenishes
    its contents from a specified serial data source when it contains
    too little data to honor a read request.

    Parameters
    ----------
    get : function
        Data retrieval function. Must return an empty sequence or None
        when it can no longer retrieve any data.
    n : int
        Number of initial entries to load into buffer.

    Methods
    -------
    clear()
        Empty buffer.
    read(n=1)
        Read `n` elements from buffer.
    replenish(n=1)
        Replenish buffer to contain at least `n` elements.

    """

    def __init__(self, get, n=1):

        if not callable(get):
            raise ValueError('get() must be callable')
        else:
            self.get = get
            self.data = []
            self.replenish(n)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

    def __iterable(self, y):
        """Check whether `y` is iterable."""

        try:
            iter(y)
        except:
            return False
        return True

    def replenish(self, n=1):
        """Attempt to replenish the buffer such that it contains at
        least `n` entries (but do not throw any exception if
        insufficient data can be obtained)."""

        while True:
            try:
                new_data = self.get()
            except:
                break
            else:

                # Append the new data to the buffer; stop attempting
                # to retrieve new data if get() doesn't return
                # anything:
                if self.__iterable(new_data):
                    if len(new_data) == 0:
                        break
                    else:
                        self.data.extend(new_data)
                else:
                    if new_data == None:
                        break
                    else:
                        self.data.append(new_data)

                if n <= len(self.data):
                    break

    def read(self, n=1):
        """Read a block of data (default length = 1)."""

        # Attempt to replenish queue if it contains too few elements:
        if n > len(self.data):
            self.replenish(n)

        # This will return without error regardless of the number of
        # entries in self.data:
        result = self.data[0:n]
        del self.data[0:n]
        return result

    def clear(self):
        """Remove all elements from the buffer."""

        self.data = []
