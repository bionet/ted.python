#!/usr/bin/env python

"""
Miscellaneous Functions
=======================

This module contains various unclassified utility functions and classes.

- chunks           Return a generator that splits a sequence into chunks.
- SerialBuffer     Buffer interface to a serial data source.

"""

__all__ = ['chunks', 'SerialBuffer']

def chunks(seq, n):
    """Return a generator whose .next() method returns length n
    subsequences of the given sequence.  If len(seq) % n != 0, the
    last subsequence returned will contain fewer than n entries."""
    
    for i in xrange(0, len(seq), n):
        yield seq[i:i+n]

class SerialBuffer:
    """This class implements a buffer that automatically replenishes
    its contents from a specified serial data source when it contains
    too little data to honor a read request."""
    
    def __init__(self, get, n=1):
        """Initialize a serial buffer with n initial entries. The data
        retrieval function get() must return an empty sequence or None
        when it can no longer retrieve any data."""

        if not callable(get):
            raise ValueError('get() must be callable')
        else:
            self.get = get
            self.data = []

    def __iterable(self, y):
        """Check whether y is iterable."""
        
        try:
            iter(y)
        except:
            return False
        return True

    def replenish(self, n=1):
        """Attempt to replenish the buffer with n entries."""

        while True:
            try:
                new_data = self.get()
            except:
                break
            else:

                # Stop retrieving new data if get() doesn't return anything:
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
