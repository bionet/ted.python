#!/usr/bin/env python

"""
Signal I/O classes
==================

Classes for writing and reading sampled and encoded signals
to and from HDF5 files.

"""

__all__ = ['ReadArray', 'WriteArray',
           'ReadSignal', 'WriteSignal',
           'ReadSampledSignal', 'WriteSampledSignal',
           'ReadTimeEncodedSignal', 'WriteTimeEncodedSignal']

import tables as t
import numpy as np

import time

class MissingDataError(AttributeError, LookupError):
    """The file does not possess a data segment."""
    pass
    
class ReadArray:    
    """A class for reading arrays of some elementary type saved in
    HDF files."""
    
    def __init__(self, filename, blocksize=0):
        """Open the specified file for reading. The file must exist. If
        no blocksize is specified, it will be set to the saved array
        length."""

        self.h5file = t.openFile(filename, 'r+')

        try:
            self.data = self.h5file.root.data
        except t.exceptions.NoSuchNodeError:
            raise MissingDataError("file `%s` does not contain any data segment" % filename)

        self.pos = 0

        if not blocksize:
            self.blocksize = len(self.data)
        else:
            if blocksize > self.data.nrows:
                print 'capping blocksize at length of signal'
                self.blocksize = len(self.data)
            elif blocksize < 0:
                raise e.ValueError('blocksize may not be negative')
            else:
                self.blocksize = blocksize

    def __del__(self):
        """Close the opened file before cleaning up."""
        
        self.close()

    def close(self):
        """Close the opened file."""

        self.h5file.close()

    def read(self):
        """Read a block of data."""

        try:
            blockdata = self.data.read(self.pos, self.pos+self.blocksize)
        except e.IndexError:
            return array((), self.data.atom.type)
        else:
            self.pos += len(blockdata)
            return blockdata

    def seek(self, offset):
        """Move data pointer to new position."""

        if offset < 0 or offset > len(self.data):
            raise e.ValueError('invalid offset')
        else:
            self.pos = offset
            
    def rewind(self):
        """Reset the file pointer to the beginning of the array."""
        
        self.pos = 0
        
class WriteArray:
    """A class for writing arrays of some elementary type to HDF
    files."""
    
    def __init__(self, filename, complevel=1, complib='lzo',
                 datatype=np.float64): 
        """Open the specified file for writing. If the specified file
        exists, it is deleted."""

        self.h5file = t.openFile(filename, 'w')

        # Initialize the data segment:
        filters = t.Filters(complevel=complevel, complib=complib)
        self.data = self.h5file.createEArray(self.h5file.root, 'data',
                                             t.Atom.from_sctype(datatype),
                                             (0, ), filters=filters)

    def __del__(self):
        """Close the opened file before cleaning up."""
        
        self.close()    

    def close(self):
        """Close the opened file."""

        self.h5file.close()
        
    def write(self, blockdata):
        """Write the specified block of data.""" 

        try:
            self.data.append(blockdata)
        except:
            raise e.IOError('error writing data')

        try:
            self.data.flush()
        except:
            raise e.IOError('error flushing data')
        
class MissingDescriptorError(AttributeError, LookupError):
    """The saved signal file does not possess a descriptor."""
    pass

class WrongDescriptorError(AttributeError, LookupError):
    """The saved signal file contains an incorrect descriptor."""
    pass

class SignalDescriptor(t.IsDescription):
    """Descriptor of saved signal."""

    comment      = t.StringCol(64, pos=1) # description of signal

SignalDescriptorDefault = ('')

class ReadSignal(ReadArray):
    """A class for reading signals stored in HDF files."""

    def __init__(self, filename, blocksize=0, descdef=SignalDescriptor):
        ReadArray.__init__(self, filename, blocksize)

        # Verify that the file contains a descriptor:
        try:
            self.descriptor = self.h5file.root.descriptor
        except t.exceptions.NoSuchNodeError:
            raise MissingDescriptorError("file `%s` does not contain a descriptor" % filename)

        # Verify that the file descriptor matches the one assumed by
        # this class:
        try:
            assert set(self.descriptor.colnames) == set(descdef.columns.keys())
        except AssertionError:
            raise WrongDescriptorError("file `%s` contains an unrecognized descriptor")
        
        # Save all of the descriptor fields in eponymous class
        # attributes:
        d = self.descriptor.read()
        for k in d.dtype.fields.keys():
            setattr(self, k, d[k][0])

class WriteSignal(WriteArray):
    """A class for writing signals to HDF files."""

    def __init__(self, filename, descvals=SignalDescriptorDefault,
                 descdef=SignalDescriptor,
                 complevel=1, complib='lzo', datatype=np.float64): 
        """Open the specified file for writing. If the specified file
        exists, it is deleted."""
        
        # Create the data segment:
        WriteArray.__init__(self, filename, complevel, complib, datatype)
        
        # Create the signal descriptor:
        self.descriptor = self.h5file.createTable(self.h5file.root,
                                                  'descriptor',
                                                  descdef,
                                                  'descriptor')
        
        # Verify that the specified descriptor can accomodate the
        # number of values that are to be stored in it:
        if descvals:            
            if len(descdef.columns) != len(descvals):
                raise ValueError("list of descriptor field values is of incorrect length")
            else:
                self.descriptor.append([descvals])
        self.descriptor.flush()
    
class SampledSignalDescriptor(t.IsDescription):
    """Descriptor of saved sampled signal."""

    comment      = t.StringCol(64, pos=1) # description of signal
    bw           = t.FloatCol(pos=2)      # bandwidth (rad/s)
    dt           = t.FloatCol(pos=3)      # interval between samples (s)

SampledSignalDescriptorDefault = ('', 1.0, 1.0)
        
class TimeEncodedSignalDescriptor(t.IsDescription):
    """Descriptor of saved time-encoded signal."""

    comment      = t.StringCol(64, pos=1) # description of signal
    bw           = t.FloatCol(pos=2)      # bandwidth (rad/s)
    dt           = t.FloatCol(pos=3)      # interval between samples (s)
    b            = t.FloatCol(pos=4)      # bias
    d            = t.FloatCol(pos=5)      # threshold
    k            = t.FloatCol(pos=6)      # integration constant
    
TimeEncodedSignalDescriptorDefault = ('', 1.0, 1.0, 1.0, 1.0, 1.0)

class ReadSampledSignal(ReadSignal):
    """A class for reading sampled signals stored in HDF files."""
    
    def __init__(self, filename, blocksize=0, descdef=SampledSignalDescriptor):
        """Open the specified file for reading. The file must exist. If
        no blocksize is specified, it will be set to the saved array
        length."""

        ReadSignal.__init__(self, filename, blocksize, descdef)
            
class WriteSampledSignal(WriteSignal):
    """A class for writing sampled signals to HDF files."""
    
    def __init__(self, filename, descvals=SampledSignalDescriptorDefault,
                 descdef=SampledSignalDescriptor,
                 complevel=1, complib='lzo', datatype=np.float64): 
        """Open the specified file for writing. If the specified file
        exists, it is deleted."""

        WriteSignal.__init__(self, filename, descvals, descdef,
                               complevel, complib, datatype)

class ReadTimeEncodedSignal(ReadSignal):
    """A class for reading time-encoded signals stored in HDF files."""
    
    def __init__(self, filename, blocksize=0,
                 descdef=TimeEncodedSignalDescriptor):
        """Open the specified file for reading. The file must exist. If
        no blocksize is specified, it will be set to the saved array
        length."""

        ReadSignal.__init__(self, filename, blocksize, descdef)
            
class WriteTimeEncodedSignal(WriteSignal):
    """A class for writing time-encoded signals to HDF files."""
    
    def __init__(self, filename, descvals=TimeEncodedSignalDescriptorDefault,
                 descdef=TimeEncodedSignalDescriptor,
                 complevel=1, complib='lzo', datatype=np.float64): 
        """Open the specified file for writing. If the specified file
        exists, it is deleted."""

        WriteSignal.__init__(self, filename, descvals, descdef, complevel,
                               complib, datatype)
