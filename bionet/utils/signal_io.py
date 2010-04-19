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
    """
    A class for reading arrays of some elementary type saved in
    an HDF5 file. More than one array may be stored in the file; the
    class assumes that each array is stored as a child of a group with
    an integer name.

    Parameters
    ----------
    filename : str
        Name of input HDF5 file.

    Methods
    -------
    close()
        Close the opened file.
    get_data_nodes()
        Retrieve the nodes of the data araays stored in the file.
    read(block_size=None, id=0)
        Read a block of data of length `block_size` from data array `id`.
    rewind(id=0)
        Reset the data pointer for data array `id` to the first entry.
    seek(offset, id=0)
        Move the data pointer for data array `id` to the indicated offset.

    """
    
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
        """Retrieve the nodes of the data arrays stored in the file."""

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
                
    def read(self, block_size=None, id=0):
        """Read a block of data from the specified data array. If no
        block size is specified, the returned block contains all data
        from the current read pointer to the end of the array. If no
        array identifier is specified, the data is read out of the
        first array."""

        if id >= len(self.data_node_list):
            raise ValueError('array id out of range')

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

    def rewind(self, id=0):
        """Reset the data pointer for the specified array to the
        beginning of the array."""
        
        self.pos[id] = 0

    def seek(self, offset, id=0):
        """Move the data pointer for the specified array to a new
        position."""

        if offset < 0 or offset > len(self.data_node_list[id].data):
            raise ValueError('invalid offset')
        else:
            self.pos[id] = offset
                    
class WriteArray:
    """
    A class for writing arrays of some elementary type to an HDF
    file. More than one array may be stored in the file; the class
    assumes that each array is stored as a child of a group with an
    integer name.

    Parameters
    ----------
    filename : str
        Output file name.
    num_arrays : int
        Number of data arrays to write to file.
    complevel : int, 0..9
        Compression level; 0 disables compression, 9 corresponds to
        maximum compression.
    complib : {'zlib', 'lzo', 'bzip2'}
        Compression filter used by pytables.
    datatype : dtype
        Data type to use in array (e.g., `numpy.float64`).

    Methods
    -------
    close()
        Close the opened file.
    get_data_nodes()
        Retrieve the nodes of the data arrays stored in the file.
    write(block_data, id=0)
        Write the specified block of data to data array `id`.

    Notes
    -----
    If the file already contains fewer data arrays than `num_arrays`,
    they will be preserved and new arrays will be initialized and
    added to the file.
    
    """
    
    def __init__(self, filename, num_arrays=1, complevel=1, complib='lzo',
                 datatype=np.float64): 

        self.h5file = t.openFile(filename, 'a')

        if num_arrays == 0:
            raise ValueError("file must contain at least one data array")

        # If the file contains fewer than the requested number of data
        # arrays, then create the requisite number of new ones:
        self.data_node_list = self.get_data_nodes()
        if len(self.data_node_list) < num_arrays:            
            filters = t.Filters(complevel=complevel, complib=complib)
            for i in xrange(len(self.data_node_list), num_arrays):
                self.__add_data(str(i), datatype, filters)
                
    def __del__(self):
        """Close the opened file before cleaning up."""
        
        self.close()    

    def __add_data(self, name, datatype, filters):
        """Add a new data array to the file."""

        group_node = self.h5file.createGroup(self.h5file.root, name)
        data_node = self.h5file.createEArray(group_node, 'data',
                                             t.Atom.from_sctype(datatype),
                                             (0, ), filters=filters)
        self.data_node_list.append(data_node)

    def __del_data(self, name):
        """Delete the specified data array in the specified group (but
        not the group itself) from the file."""

        self.h5file.removeNode(self.h5file.root, '/' + name + '/data')
    
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
        """Write the specified block of data to the specified data array.""" 

        if id >= len(self.data_node_list):
            raise ValueError('array id out of range')
        
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

def get_desc_defaults(desc):
    """Extract the default column values from a descriptor class.

    Parameters
    ----------
    desc : subclass of `tables.IsDescription`
       Descriptor class.

    Returns
    -------
    vals : list
       List of default column values.
       
    See Also
    --------
    tables.IsDescription
    
    """

    if not issubclass(desc, t.IsDescription):
        raise ValueError("argument is not a descriptor class")

    vals = []
    for key in desc.columns.keys():
        vals.append(desc.columns[key].dflt)
    return vals

def get_desc_types(desc):
    """Extract the dtypes of the columns of a descriptor class.

    Parameters
    ----------
    desc : subclass of `tables.IsDescription`
       Descriptor class.

    Returns
    -------
    types : list
       List of column dtypes.

    See Also
    --------
    tables.IsDescription

    """

    if not issubclass(desc, t.IsDescription):
        raise ValueError("argument is not a descriptor class")

    types = []
    for key in desc.columns.keys():
        types.append(desc.columns[key].dtype)
    return types

class ReadSignal(ReadArray):
    """
    A class for reading signals stored in HDF5 files. A single file
    may contain multiple signals. Each signal contains a data array
    and a descriptor.

    Parameters
    ----------
    filename : str
        Input file name.

    Methods
    -------
    close()
        Close the opened file.
    get_data_nodes()
        Retrieve the nodes of the data arrays stored in the file.
    get_desc_nodes()
        Retrieve the descriptor nodes of the data arrays stored in
        the file.
    read(block_size=None, id=0)
        Read a block of data of length `block_size` from data array `id`.
    read_desc(id=0)
        Return the data in the descriptor of data array `id`.
    rewind(id=0)
        Reset the data pointer for data array `id` to the first entry.
    seek(offset, id=0)
        Move the data pointer for data array `id` to the indicated offset.

    """
    
    def __init__(self, filename):
        ReadArray.__init__(self, filename)

        # Retrieve the data descriptors:
        self.desc_node_list = self.get_desc_nodes()
        if len(self.data_node_list) != len(self.desc_node_list):
            raise DescriptorDataMismatchError("file `%s` contains " +
                                              "differing numbers of descriptors and data arrays" % filename)

        # Validate the descriptors:
        self.__validate_descs()
        
    def __validate_descs(self):
        """Validate the signal descriptors in the file. This method
        may be implemented in subclasses as necessary."""

        pass
    
    def read_desc(self, id=0):
        """Return the data in the specified data descriptor as a list
        of values."""

        return self.desc_node_list[id].read()[0]
    
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
    """A class for writing signals to an HDF file. More than one
    signal may be stored in the file; the class assumes that each
    array is stored as a child of a group with an integer name.

    Parameters
    ----------
    filename : str
        Output file name.
    desc_vals : list of lists
        Default descriptor values. Each descriptor's default values
        must be specified as a separate list.
    desc_defs : list of descriptor classes
        Descriptor classes. Each class must be a child of
        `tables.IsDescription`.
    complevel : int, 0..9
        Compression level; 0 disables compression, 9 corresponds to
        maximum compression.
    complib : {'zlib', 'lzo', 'bzip2'}
        Compression filter used by pytables.
    datatype : dtype
        Data type to use in array (e.g., `numpy.float64`).

    Methods
    -------
    close()
        Close the opened file.
    get_data_nodes()
        Retrieve the nodes of the data arrays stored in the file.
    get_desc_nodes()
        Retrieve the descriptor nodes of the data arrays stored in
        the file.
    write(block_data, id=0)
        Write the specified block of data to data array `id`.

    Notes
    -----
    If the file already contains fewer data arrays than `num_arrays`,
    they will be preserved and new arrays will be initialized and
    added to the file.

    """

    def __init__(self, filename,
                 desc_vals=[get_desc_defaults(SignalDescriptor)],
                 desc_defs=[SignalDescriptor],
                 complevel=1, complib='lzo', datatype=np.float64): 
        """Open the specified file for writing. If the file already
        contains data arrays, new arrays are added to bring the total
        number up to the number of specified signal descriptors."""
        
        # Make sure each data segment has a descriptor:
        if len(desc_vals) != len(desc_defs):
            raise ValueError('number of descriptor definitions does ' +
                             'not equal the number of descriptor data tuples')

        # Validate the descriptors:
        self.__validate_descs(desc_vals, desc_defs)
        
        # Create the data arrays:
        WriteArray.__init__(self, filename, len(desc_vals),
                            complevel, complib, datatype)
        
        # When the number of specified descriptors exceeds the number
        # actually in the file..
        self.desc_node_list = self.get_desc_nodes()
        if len(self.desc_node_list) < len(desc_vals):

            # Remove any existing descriptors so that they can be
            # replaced by the specified descriptors:
            for i in xrange(len(self.desc_node_list)):
                self.h5file.removeNode(self.h5file.root,
                                       '/' + str(i) + '/descriptor')
                                       
            # Create descriptors for the data segments:
            for i in xrange(len(desc_vals)):
                self.__add_desc(str(i), desc_defs[i], desc_vals[i])

    def __validate_descs(self, desc_vals, desc_defs):
        """Validate the specified signal descriptors. This method
        may be implemented in subclasses as necessary."""

        pass
    
    def __add_desc(self, name, desc_def, desc_val):
        """Add a new descriptor to the array in the specified group."""

        desc_node = \
                  self.h5file.createTable(self.h5file.root.__getattr__(name),
                                          'descriptor', desc_def, 'descriptor')

        # Verify that the specified descriptor can accomodate the
        # number of values that are to be stored in it:
        if len(desc_def.columns) != len(desc_val):
            raise ValueError("list of descriptor field values " +
                             "is of incorrect length")
        else:
            desc_node.append([tuple(desc_val)])
            desc_node.flush()
            self.desc_node_list.append(desc_node)
        
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

    comment      = t.StringCol(64, pos=1)      # description of signal
    bw           = t.FloatCol(pos=2, dflt=1.0) # bandwidth (rad/s)
    dt           = t.FloatCol(pos=3, dflt=1.0) # interval between samples (s)

class TimeEncodedSignalDescriptor(t.IsDescription):
    """Descriptor of saved time-encoded signal."""

    comment      = t.StringCol(64, pos=1)      # description of signal
    bw           = t.FloatCol(pos=2, dflt=1.0) # bandwidth (rad/s)
    dt           = t.FloatCol(pos=3, dflt=1.0) # interval between samples (s)
    b            = t.FloatCol(pos=4, dflt=1.0) # bias
    d            = t.FloatCol(pos=5, dflt=1.0) # threshold
    k            = t.FloatCol(pos=6, dflt=1.0) # integration constant
    
class ReadSampledSignal(ReadSignal):
    """A class for reading sampled signals stored in HDF5 files."""

    def __validate_descs(self):
        """Validate the descriptors in the file by making sure that
        they possess the same columns as the SampledSignalDescriptor
        class."""

        for desc_node in self.desc_node_list:
            try:
                assert set(desc_node.colnames) == \
                       set(SampledSignalDescriptor.columns.keys())
            except AssertionError:
                raise WrongDescriptorError("file `%s` contains " +
                                           "an unrecognized descriptor" % filename)

class WriteSampledSignal(WriteSignal):
    """A class for writing sampled signals to HDF5 files."""

    def __init__(self, filename, 
                 desc_vals=[get_desc_defaults(SampledSignalDescriptor)],
                 complevel=1, complib='lzo', datatype=np.float64): 
        """Open the specified file for writing. If the file already
        contains data arrays, new arrays are added to bring the total
        number up to the number of specified signal descriptors. """

        WriteSignal.__init__(self, filename, desc_vals,
                             [SampledSignalDescriptor]*len(desc_vals),
                             complevel, complib, datatype)

    def __validate_descs(self, desc_vals, desc_defs):
        """Validate the specified signal descriptors and values by
        making sure that they agree."""

        for (desc_val, desc_def) in zip(desc_vals, desc_defs):
            if map(type, desc_val) != get_desc_types(desc_def):
                raise WrongDescriptorError("descriptor values do not match format")

class ReadTimeEncodedSignal(ReadSignal):
    """A class for reading time-encoded signals stored in HDF5 files."""

    def __validate_descs(self):
        """Validate the descriptors in the file by making sure that
        they possess the same columns as the
        TimeEncodedSignalDescriptor class."""

        for desc_node in self.desc_node_list:
            try:
                assert set(desc_node.colnames) == \
                       set(TimeEncodedSignalDescriptor.columns.keys())
            except AssertionError:
                raise WrongDescriptorError("file `%s` contains " +
                                           "an unrecognized descriptor" % filename)
            
class WriteTimeEncodedSignal(WriteSignal):
    """A class for writing time-encoded signals to HDF5 files."""
    
    def __init__(self, filename, 
                 desc_vals=[get_desc_defaults(TimeEncodedSignalDescriptor)],
                 complevel=1, complib='lzo', datatype=np.float64): 
        """Open the specified file for writing. If the file already
        contains data arrays, new arrays are added to bring the total
        number up to the number of specified signal descriptors. """

        WriteSignal.__init__(self, filename, desc_vals,
                             [TimeEncodedSignalDescriptor]*len(desc_vals),
                             complevel, complib, datatype)

    def __validate_descs(self, desc_vals, desc_defs):
        """Validate the specified signal descriptors and values by
        making sure that they agree."""

        for (desc_val, desc_def) in zip(desc_vals, desc_defs):
            if map(type, desc_val) != get_desc_types(desc_def):
                raise WrongDescriptorError("descriptor values do not match format")

if __name__ == '__main__':

    # Short demo of how to use the above classes:
    from os import remove
    from tempfile import mktemp
    from atexit import register

    # Write to a file:
    file_name = mktemp() + '.h5'
    N = 1000
    x1 = np.random.rand(N)
    x2 = np.random.rand(N)
    w = WriteArray(file_name, 2)
    w.write(x1)
    w.write(x2,id=1)
    w.close()

    # Clean up on exit:
    register(remove, file_name)

    # Read the data from the file:
    r = ReadArray(file_name)
    y1 = r.read()
    y2 = r.read(id=1)
    r.close()
    
    assert np.all(x1 == y1)
    assert np.all(x2 == y2)

