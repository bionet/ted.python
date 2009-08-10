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
import warnings as w
import time

# Suppress warnings provoked by the use of integers as HDF5 group names:
w.simplefilter('ignore', t.NaturalNameWarning)

class MissingDataError(AttributeError, LookupError):
    """The file does not possess any data segments."""
    pass
    
class ReadArray:    
    """A class for reading arrays of some elementary type saved in
    an HDF file. More than one array may be stored in the file; the
    class assumes that each array is stored as a child of a group with
    an integer name."""
    
    def __init__(self, filename, *args):
        """Open the specified file for reading."""

        self.h5file = t.openFile(filename, 'r+')

        # Retrieve the nodes corresponding to the data arrays:
        self.data_node_list = self.get_data_nodes()
        num_arrays = len(self.data_node_list)        
        if num_arrays == 0:
            raise MissingDataError("file `%s` does not contain any data segments" % filename)

        # Initialize read pointers:
        self.pos = np.zeros(num_arrays, int)
            
    def __del__(self):
        """Close the opened file before cleaning up."""
        
        self.close()

    def close(self):
        """Close the opened file."""

        self.h5file.close()

    def get_data_nodes(self):
        """Retrieve the nodes of the data arrays stored within the file."""

        # Each array must be stored as
        # self.h5file.root.ARRAY_NAME.data, where ARRAY_NAME is an
        # integer:
        data_node_list = []
        for n in self.h5file.root:
            try:
                int(n._v_name)
            except ValueError:
                raise ValueError('file contains non-integer data name')

            try:
                node = n.data
            except t.exceptions.NoSuchNodeError:
                pass
            else:
                data_node_list.append(node)
        return data_node_list
                
    def read(self, id=0, block_size=None):
        """Read a block of data from the specified data array. If no
        block size is specified, the returned block contains all data
        from the current read pointer to the end of the array. If no
        array identifier is specified, the data is read out of the
        first array."""

        g = self.data_node_list[id]

        try:
            if block_size == None:
                block_data = g.read(self.pos[id], len(g))
            else:
                block_data = g.read(self.pos[id],
                                    self.pos[id]+block_size)
        except IndexError:
            return array((), g.atom.type)
        else:
            self.pos[id] += len(block_data)
            return block_data

    def seek(self, offset, id=0):
        """Move the data pointer for the specified array to a new
        position."""

        if offset < 0 or offset > len(self.data_node_list[id].data):
            raise ValueError('invalid offset')
        else:
            self.pos[id] = offset
            
    def rewind(self, id=0):
        """Reset the data pointer for the specified array to the
        beginning of the array."""
        
        self.pos[id] = 0
        
class WriteArray:
    """A class for writing arrays of some elementary type to an HDF
    file. More than one array may be stored in the file."""
    
    def __init__(self, filename, num_arrays=1, complevel=1, complib='lzo',
                 datatype=np.float64): 
        """Open the specified file for writing. If the file already
        contains data arrays, new arrays are added to bring the total
        number up to num_arrays. """

        self.h5file = t.openFile(filename, 'a')

        if num_arrays == 0:
            raise ValueError("file must contain at least one data array")

        # If the file contains fewer than the requested number of data
        # arrays, then create the requisite number of new ones:
        self.data_node_list = self.get_data_nodes()
        if len(self.data_node_list) < num_arrays:            
            filters = t.Filters(complevel=complevel, complib=complib)
            for i in xrange(len(self.data_node_list), num_arrays):
                group_node = self.h5file.createGroup(self.h5file.root, str(i))
                data_node = self.h5file.createEArray(group_node, 'data',
                                             t.Atom.from_sctype(datatype),
                                             (0, ), filters=filters)
                self.data_node_list.append(data_node)
                
    def __del__(self):
        """Close the opened file before cleaning up."""
        
        self.close()    

    def close(self):
        """Close the opened file."""

        self.h5file.close()

    def get_data_nodes(self):
        """Retrieve the data array nodes stored within the file."""

        # Each array must be stored as
        # self.h5file.root.ARRAY_NAME.data, where ARRAY_NAME is an
        # integer:
        data_node_list = []
        for n in self.h5file.root:
            try:
                int(n._v_name)
            except ValueError:
                raise ValueError('file contains non-integer data name')

            try:
                node = n.data
            except t.exceptions.NoSuchNodeError:
                pass
            else:
                data_node_list.append(node)
        return data_node_list

    def write(self, block_data, id=0):
        """Write the specified block of data.""" 

        try:
            self.data_node_list[id].append(block_data)
        except:
            raise IOError('error writing data')

        try:
            self.data_node_list[id].flush()
        except:
            raise IOError('error flushing data')
        
class MissingDescriptorError(AttributeError, LookupError):
    """The saved signal file does not possess a descriptor."""
    pass

class WrongDescriptorError(AttributeError, LookupError):
    """The saved signal file contains an incorrect descriptor."""
    pass

class DescriptorDataMismatchError(AttributeError, LookupError):
    """The number of descriptors in the saved signal file differs from
    the number of data arrays."""
    pass

class SignalDescriptor(t.IsDescription):
    """Descriptor of saved signal."""

    comment      = t.StringCol(64, pos=1) # description of signal

SignalDescriptorDefault = ('',)
    
class ReadSignal(ReadArray):
    """A class for reading signals stored in HDF files."""

    def __init__(self, filename, desc_defs=[SignalDescriptor]):
        """Open the specified file for reading."""
        
        ReadArray.__init__(self, filename)

        # Retrieve the nodes corresponding to the data descriptors:
        self.desc_node_list = self.get_desc_nodes()
        if len(self.data_node_list) != len(self.desc_node_list):
            raise DescriptorDataMismatchError("file `%s` contains " +
                                              "differing numbers of descriptors and data arrays" % filename)

        self.descriptors = []
        for (node,i) in zip(self.desc_node_list,
                            xrange(len(self.desc_node_list))):

            # Verify that the descriptors matches the one assumed by this
            # class:
            try:
                assert set(node.colnames) == set(desc_defs[i].columns.keys())
            except AssertionError:
                raise WrongDescriptorError("file `%s` contains " +
                                           "an unrecognized descriptor" % filename)

    def read_desc(self, id=0):
        """Return the data in the specified data descriptor."""

        return self.desc_node_list[id]
    
    def get_desc_nodes(self):
        """Retrieve the signal descriptors stored within the file."""

        # Each descriptor must be stored as
        # self.h5file.root.ARRAY_NAME.descriptor, where ARRAY_NAME is an
        # integer:
        desc_node_list = []
        for n in self.h5file.root:
            try:
                int(n._v_name)
            except ValueError:
                raise ValueError('file contains non-integer data name')

            try:
                node = n.descriptor
            except t.exceptions.NoSuchNodeError:
                pass
            else:
                desc_node_list.append(node)
        return desc_node_list

class WriteSignal(WriteArray):
    """A class for writing signals to HDF files."""

    def __init__(self, filename, num_arrays=1,
                 desc_vals=[SignalDescriptorDefault],
                 desc_defs=[SignalDescriptor],
                 complevel=1, complib='lzo', datatype=np.float64): 
        """Open the specified file for writing. If the file already
        contains data arrays, new arrays are added to bring the total
        number up to num_arrays. """
        
        # Create the data arrays:
        WriteArray.__init__(self, filename, num_arrays,
                            complevel, complib, datatype)

        # Create the signal descriptors:
        if len(desc_vals) != len(desc_defs):
            raise ValueError('number of descriptor definitions does ' +
                             'not equal the number of descriptor data tuples')
        self.desc_node_list = self.get_desc_nodes()
        if len(self.desc_node_list) < num_arrays:
            for i in xrange(len(self.desc_node_list), num_arrays):
                node = self.h5file.createTable(self.h5file.root.__getattr__(str(i)),
                                               'descriptor',
                                               desc_defs[i],
                                               'descriptor')
        
                # Verify that the specified descriptor can accomodate the
                # number of values that are to be stored in it:
                if len(desc_defs[i].columns) != len(desc_vals[i]):
                    raise ValueError('list of descriptor field values ' +
                                     'is of incorrect length')
                else:
                    node.append([desc_vals[i]])
                    node.flush()
                    self.desc_node_list.append(node)

    def get_desc_nodes(self):
        """Retrieve the signal descriptors stored within the file."""

        # Each descriptor must be stored as
        # self.h5file.root.ARRAY_NAME.descriptor, where ARRAY_NAME is an
        # integer:
        desc_node_list = []
        for n in self.h5file.root:
            try:
                int(n._v_name)
            except ValueError:
                raise ValueError('file contains non-integer data name')

            try:
                node = n.descriptor
            except t.exceptions.NoSuchNodeError:
                pass
            else:
                desc_node_list.append(node)
        return desc_node_list

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
    
    def __init__(self, filename, num_arrays=1,
                 desc_defs=[SampledSignalDescriptor]):
        """Open the specified file for reading."""

        ReadSignal.__init__(self, filename, num_arrays, desc_defs)
            
class WriteSampledSignal(WriteSignal):
    """A class for writing sampled signals to HDF files."""

    def __init__(self, filename, num_arrays=1,
                 desc_vals=[SampledSignalDescriptorDefault],
                 desc_defs=[SampledSignalDescriptor],
                 complevel=1, complib='lzo', datatype=np.float64): 
        """Open the specified file for writing. If the file already
        contains data arrays, new arrays are added to bring the total
        number up to num_arrays. """

        WriteSignal.__init__(self, filename, num_arrays, desc_vals,
                               desc_defs, complevel, complib,
                               datatype)

class ReadTimeEncodedSignal(ReadSignal):
    """A class for reading time-encoded signals stored in HDF files."""
    
    def __init__(self, filename, num_arrays=1,
                 desc_defs=[TimeEncodedSignalDescriptor]):
        """Open the specified file for reading."""

        ReadSignal.__init__(self, filename, num_arrays, desc_defs)
            
class WriteTimeEncodedSignal(WriteSignal):
    """A class for writing time-encoded signals to HDF files."""
    
    def __init__(self, filename, num_arrays=1,
                 desc_vals=[TimeEncodedSignalDescriptorDefault],
                 desc_defs=[TimeEncodedSignalDescriptor],
                 complevel=1, complib='lzo', datatype=np.float64): 
        """Open the specified file for writing. If the file already
        contains data arrays, new arrays are added to bring the total
        number up to num_arrays. """

        WriteSignal.__init__(self, filename, num_arrays, desc_vals,
                               desc_defs, complevel, complib,
                               datatype)
