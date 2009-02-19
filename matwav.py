#!/usr/bin/env python

"""
Classes for reading and writing WAV files using the normalization conventions
employed by Matlab.

NOTE: These classes can only read 8-bit WAV files.
"""

import wave                   
import struct

from numpy import array,float

class ReadMatWav(wave.Wave_read):
    """An extension of the wave.Wave_read class that provides for the
    reading of WAV files in blocks using the normalization conventions
    employed by Matlab."""

    def __init__(self,filename,blocksize=0,overlap=0):
        """Open a WAV file for reading with the specified blocksize
        and block overlap. If no blocksize is specified, the blocksize is
        set to the length of the signal. If overlap > 0, each block
        read will overlap the previous block by the specified number
        of elements."""

        if type(blocksize) is not int:
            raise ValueError('blocksize must be an integer')
        if type(overlap) is not int:
            raise ValueError('overlap must be an integer')
        if overlap and overlap >= blocksize/2:
            raise ValueError('overlap may not exceed half the blocksize')
        
        wave.Wave_read.__init__(self,filename)

        if self.getsampwidth() != 1 or self.getnchannels() != 1:
            raise ValueError('input file must be in 8 mono format')
        
        if not blocksize:
            self.blocksize = self.getnframes()
        else:
            self.blocksize = blocksize
        self.overlap = overlap

    def readblock(self):
        """Read a block of data from the open WAV file and normalize it
        according to Matlab conventions."""

        # Return an empty array when the end of the signal has been reached:
        if self._soundpos == self._nframes:
            return array(())

        # Decrement the sound pointer by the overlap once it has been
        # advanced sufficiently:
        if self._soundpos > self.overlap:
            self.setpos(self._soundpos-self.overlap)
        data = self.readframes(self.blocksize)
        return (array(struct.unpack("%dB" % len(data),
                                    data),float)-128.0)/128.0
        
class WriteMatWav(wave.Wave_write):
    """An extension of the wave.Wave_write class that provides for the
    writing of WAV files in blocks using the normalization conventions
    employed by Matlab."""

    def __init__(self,filename,params=(1,1,22050,0,'NONE','not compressed')):
        """Open a WAV file for writing."""
        
        wave.Wave_write.__init__(self,filename)
        self.setparams(params)
        
    def writeblock(self,data):
        """Write a block of data to the open WAV file. The data is assumed
        to have been normalized according to Matlab conventions."""

        # Reverse the normalization of the data and pack it before writing:
        self.writeframes(struct.pack("%dB" %
                                     len(data),*array(128*(data+1),int)))
