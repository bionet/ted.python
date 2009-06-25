#!/usr/bin/env python

"""
Miscellaneous Functions
=======================

This module contains various unclassified utility functions.

- chunks           Return a generator that splits a sequence into chunks.

"""

__all__ = ['chunks']

def chunks(seq, n):
    """Return a generator whose .next() method returns length n
    subsequences of the given sequence.  If len(seq) % n != 0, the
    last subsequence returned will contain fewer than n entries."""
    
    for i in xrange(0, len(seq), n):
        yield seq[i:i+n]

