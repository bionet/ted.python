#!/usr/bin/env python

"""
Real-time time encoding and decoding algorithms.
"""

__all__ = ['RealTimeEncoder', 'RealTimeDecoder', 'RealTimeDecoderIns',
           'asdm_encode_real','asdm_decode_real','asdm_decode_ins_real']

from numpy import arange, sin, cos, pi, zeros, array, floor, ceil, \
     float, cumsum, round, hstack, intersect1d, where
import bionet.ted.asdm as asdm
import bionet.ted.vtdm as vtdm
import bionet.utils.misc as misc

class AbstractSignalProcessor:
    """This abstract class describes a signal processor that retrieves
    blocks of data from a source, applies some processing algorithm to
    it, and saves the processed blocks. It must be subclassed in order
    to be made functional."""
    
    def __init__(self, get, put, verbose=True):
        """Initialize a signal processor.

        Parameters
        ----------
        get: function that returns a numpy array
            Used to retrieve data to process; should return an empty array
            when no more data is available to process.
        put: function that accepts a numpy array
            Used to save processed data.
        verbose: bool
            Diagnostic messages are printed when various class methods
            are invoked.
            
        Notes
        -----
        The process() method of the signal processor may be called immediately
        after initialization.
        """

        # Make sure that the specified data retrieval and storage routines are
        # actually functions:
        if not callable(get):
            raise ValueError('get() must be callable')
        if not callable(put):
            raise ValueError('put() must be callable')

        self.get = get
        self.put = put

        self.verbose = verbose

        # This should be set once the process() method will not (or
        # should not) return anything:
        self.done = False
        
    def process(self, block):
        """Process the specified block of data and return a block of
        processed data. In general, the returned data might not necessarily
        correspond to the input. This method knows how to respond to empty
        blocks.

        Parameters
        ----------
        block: numpy array
            Block of data to process.
        """

        pass

    def run(self):
        """Process data until the retrieval routine returns no data when
        invoked."""

        count = 0
        while True:
            block = self.get()
            # try:
            #     block = self.get()
            # except:
            #     raise IOError('error reading input data')
            
            self.put(self.process(block))

            # This should be the only way the processing loop terminates:
            if self.done:
                break
            
            if self.verbose:
                print 'block %i' % count
            count += 1
            
class RealTimeEncoder(AbstractSignalProcessor):
    """This class implements a real-time time encoding machine."""

    def __init__(self, get, put, dt, b, d, k=1.0, dte=0.0, verbose=True):
        """Initialize a real-time time encoder.

        Parameters
        ----------
        get: function that returns a numpy array
            Used to retrieve data to process; should return an empty array
            when no more data is available to process.
        put: function that accepts a numpy array
            Used to save processed data.
        dt: float
            Sampling resolution of input signal; the sampling frequency
            is 1/dt Hz.
        b: float
            Encoder bias.
        d: float
            Encoder threshold.
        k: float
            Encoder integration constant.
        dte: float
            Sampling resolution assumed by the encoder. This may not exceed
            dt.
        """

        AbstractSignalProcessor.__init__(self, get, put, verbose)
        self.dt = dt
        self.b = b
        self.d = d
        self.k = k
        self.dte = dte

        self.y = 0.0
        self.interval = 0.0
        self.sgn = 1

    def process(self, block):
        """Encode the specified block of data and return the result.

        Parameters
        ----------
        block: numpy array
            Block of data to encode.
        """

        s, self.y, self.interval, self.sgn = \
           asdm.asdm_encode(block, self.dt, self.b, self.d,
                            self.k, self.dte, self.y, self.interval,
                            self.sgn, 'trapz', True)
        if not len(block):
            self.done = True
        return s
        
class AbstractRealTimeDecoder(AbstractSignalProcessor):
    """This class implements a real-time time decoding machine. It
    must be subclassed to provide it with a functional block decoding
    algorithm."""

    def __init__(self, get, put, dt, bw, N, M, K, verbose=True):
        """Initialize a real-time time decoder.

        Parameters
        ----------
        get: function that returns a numpy array
            Used to retrieve data to process; should return an empty array
            when no more data is available to process.
        put: function that accepts a numpy array
            Used to save processed data.
        dt: float
            Sampling resolution of input signal; the sampling frequency
            is 1/dt Hz.
        bw: float
            Signal bandwidth (in rad/s).
        N: int
            Number of spikes to process in each block less 1.
        M: int
            Number of spikes between the starting time of each successive
            block.
        K: int
            Number of spikes in the overlap between successive blocks.   
        """

        AbstractSignalProcessor.__init__(self, get, put, verbose)

        if N <= 1:
            raise ValueError('N must exceed 1')
        if M <= 0 or M > N/2:
            raise ValueError('M must be in the range (0,N/2)')
        if K < 1 or K >= N-2*M:
            raise ValueError('K must be in the range [1,N-2*M)') # ???
                    
        self.dt = dt
        self.bw = bw
        self.N = N
        self.M = M
        self.K = K
        self.J = N-2*M-K # number of spikes between overlapping blocks

        # Needed to adjust sign for compensation principle used in
        # decoding algorithm:
        self.first_spike = 1

        # Spike intervals and spike indicies:
        self.s = []
        self.tk = array((), float)

        # Overlap:
        self.overlap = array((), float)
        
        # Number of spike intervals that must be obtainable from the
        # queue. For the first block, N+2 spike intervals must be
        # retrieved because the last spike is discarded during the
        # reconstruction:
        self.intervals_needed = self.N+2

        # When these flags are set, the block of data being decoded is
        # windowed on its left and right sides as indicated. The
        # following initial values are set because the very first
        # block does not need to stitched to any other block:
        self.window_left = False
        self.window_right = True

        # Since the number of spike intervals in a block of data may
        # vary, the encoded data is queued so that the decoding
        # algorithm can extract specific numbers of spike intervals to
        # process as needed:
        self.queue = []

    def decode_block(self, block):
        """Decode a block of data. This method must be implemented by subclasses
        of the abstract class."""

        pass

    def process(self, block):
        """Insert the specified block of data into the decoder queue
        and return a block of decoded data.

        Parameters
        ----------
        block: numpy array
            Block of data to encode.
        """

        # Processing has already been completed:
        if self.done:
            return array((), float)
        
        # Append the data to the queue:
        self.queue.extend(block)

        # If the input block is not empty but there is not enough data
        # in the queue to process a full block of N+1 spikes, return
        # an empty array because more data must be added to the queue
        # before processing may continue. If the input block is empty
        # and the number of spike intervals in the queue is still
        # lower than the number needed for an overlapping window after
        # adding the specified block of input data to the queue, the
        # final block has been reached and hence should not be
        # windowed on its right side:
        if len(self.queue) < self.intervals_needed:
            if len(block):
                return array((), float)
            else:
                self.window_right = False

        # Append new spike intervals to the current block. After
        # the first block, drop the first J spike intervals in the
        # block and append J new intervals:
        if self.intervals_needed < len(self.queue):
            self.intervals_to_add = self.queue[0:self.intervals_needed]
            del self.queue[0:self.intervals_needed]
        else:
            self.intervals_to_add = self.queue[:]
            del self.queue[:]
        self.s.extend(self.intervals_to_add)

        if self.intervals_needed != self.J:
            self.intervals_needed = self.J
        else:
            del self.s[0:self.J]

        # Find the times of the spikes in the current block:
        self.ts = cumsum(self.s)
        self.tk = array(round(self.ts/self.dt), int)
        self.curr_dur = max(self.ts)
        self.t = arange(0, self.curr_dur, self.dt)

        # Decode the current block:
        self.u = self.decode_block(self.s)
            
        # Discard the portion of the reconstructed signal after
        # the second to last spike interval for all blocks except the
        # last one:
        if self.window_right:
            self.n = self.tk[-1]
            self.tk = self.tk[0:-1]
            self.u = self.u[0:self.n]
            self.t = self.t[0:self.n]

        # The sign of the spike at the beginning of the next block
        # must be the reverse of the current one if J is odd:
        if self.J % 2:
            self.first_spike *= -1

        # Construct and apply shaping window to decoded signal:
        if self.window_left:
            ll = self.ts[self.M]
            lr = self.ts[self.M+self.K]
        else:
            ll = -self.dt # needed to force first entry in window to be 1
            lr = 0.0
        if self.window_right:
            rl = self.ts[self.N-self.M-self.K]
            rr = self.ts[self.N-self.M]
        else:
            rl = self.t[-1]
            rr = self.t[-1]
        self.w = self.window(self.t, ll, lr, rl, rr)
        self.uw = self.u*self.w
            
        # Apart from the first block, the saved nonzero
        # overlapping portion of the previous block must be
        # combined with that of the current block:
        if self.window_left:
            self.u_out = self.overlap+\
                         self.uw[self.tk[self.M]:self.tk[self.M+self.K]]
        else:
            self.u_out = self.uw[0:self.tk[self.M+self.K]]

        # Apart from the last block, the nonzero portion of the
        # current block that will overlap with the next block must
        # be retained for the next iteration:
        if self.window_right:
            self.u_out = hstack((self.u_out,
                self.uw[self.tk[self.M+self.K]:self.tk[self.N-self.M-self.K]]))
            self.overlap = \
                self.uw[self.tk[self.N-self.M-self.K]:self.tk[self.N-self.M]]
        else:
            self.u_out = hstack((self.u_out, self.uw[self.tk[self.M+self.K]::]))
            self.overlap = array((), float)
            
        # The first block decoded should only be windowed on its
        # right side if at all. Hence, if window_left is false and
        # window_right is true, window_left should be set to true
        # before the next block is processed so that it is
        # windowed on its left side:
        if self.window_right and not self.window_left:
            self.window_left = True
            
        # If window_left is true and window_right is false, the
        # last block has been decoded:
        if not self.window_right:
            self.done = True

        # Return current decoded block:
        return self.u_out

    # Methods for computing the edges of the windows:
    def _theta1(self, t, l, r):
        return sin((pi/2)*(t-l)/(r-l))**2

    def _theta2(self, t, l, r):
        return cos((pi/2)*(t-l)/(r-l))**2

    ### NOTE: this method sometimes produces NaN entries and should
    ### therefore not be used:
    def window_bad(self, t, ll, lr, rl, rr):
        '''Return a window defined over the vector of times t that
        forms a partition of unity over all time. The function is
        equal to 0 when t <= ll or t > rr, theta(t,ll,lr) when ll < t
        <= lr, 1 when lr < t <= rl, and 1-theta(t,rl,rr)
        when rl < t <= rr.'''
    
        n = len(t)
        w = zeros(n, float)
        n = float(n)
        m = max(t)

        w[int(floor(ll*n/m)):int(ceil(lr*n/m))] = \
              self._theta1(t[int(floor(ll*n/m)):int(ceil(lr*n/m))], ll, lr)
        w[int(ceil(lr*n/m)):int(floor(rl*n/m))] = 1.0
        w[int(floor(rl*n/m)):int(ceil(rr*n/m))] = \
              self._theta2(t[int(floor(rl*n/m)):int(ceil(rr*n/m))], rl, rr)
        
        return w

    def window(self, t, ll, lr, rl, rr):
        '''Return a window defined over the vector of times t that
        forms a partition of unity over all time. The function is
        equal to 0 when t <= ll or t > rr, theta(t,ll,lr) when ll < t
        <= lr, 1 when lr < t <= rl, and 1-theta(t,rl,rr)
        when rl < t <= rr.'''

        w = zeros(len(t), float)
        
        i1 = intersect1d(where(ll < t)[0], where(t <= lr)[0])
        i2 = intersect1d(where(lr < t)[0], where(t <= rl)[0])
        i3 = intersect1d(where(rl < t)[0], where(t <= rr)[0])
        
        w[i1] = self._theta1(t[i1], ll, lr)
        w[i2] = 1.0
        w[i3] = self._theta2(t[i3], rl, rr)
        
        return w

class RealTimeDecoder(AbstractRealTimeDecoder):
    """This class implements a real-time time decoding machine."""

    def __init__(self, get, put, dt, bw, b, d, k, N, M, K, verbose=True):
        """Initialize a real-time time decoder.

        Parameters
        ----------
        get: function that returns a numpy array
            Used to retrieve data to process; should return an empty array
            when no more data is available to process.
        put: function that accepts a numpy array
            Used to save processed data.
        dt: float
            Sampling resolution of input signal; the sampling frequency
            is 1/dt Hz.
        bw: float
            Signal bandwidth (in rad/s).
        b: float
            Encoder bias.
        d: float
            Decoder threshold.
        k: float
            Decoder integration constant.    
        N: int
            Number of spikes to process in each block less 1.
        M: int
            Number of spikes between the starting time of each successive
            block.
        K: int
            Number of spikes in the overlap between successive blocks.   

        """

        AbstractRealTimeDecoder.__init__(self, get, put, dt, bw, \
                                         N, M, K, verbose)
        self.b = b
        self.d = d
        self.k = k

    def decode_block(self, block):
        """Decode a block of data."""

        return vtdm.vander_decode(block, self.curr_dur, self.dt, self.bw,
                                  self.b, self.d, self.k, self.first_spike)

class RealTimeDecoderIns(AbstractRealTimeDecoder):
    """This class implements a parameter-insensitive real-time time
    decoding machine."""

    def __init__(self, get, put, dt, bw, b, N, M, K, verbose=True):
        """Initialize a real-time time decoder.

        Parameters
        ----------
        get: function that returns a numpy array
            Used to retrieve data to process; should return an empty array
            when no more data is available to process.
        put: function that accepts a numpy array
            Used to save processed data.
        dt: float
            Sampling resolution of input signal; the sampling frequency
            is 1/dt Hz.
        bw: float
            Signal bandwidth (in rad/s).
        b: float
            Encoder bias.
        N: int
            Number of spikes to process in each block less 1.
        M: int
            Number of spikes between the starting time of each successive
            block.
        K: int
            Number of spikes in the overlap between successive blocks.   
        """

        AbstractRealTimeDecoder.__init__(self, get, put, dt, bw, \
                                         N, M, K, verbose)
        self.b = b

    def decode_block(self, block):
        """Decode a block of data."""

        return vtdm.vander_decode_ins(block, self.curr_dur, self.dt, self.bw,
                                      self.b, self.first_spike)

def asdm_encode_real(u, dt, b, d, k=1.0, dte=0.0):
    """Encode an arbitrarily long signal using an asynchronous
    sigma-delta modulator.

    Parameters
    ----------
    u: numpy array of floats
        Signal to encode.
    dt: float
        Sampling resolution of input signal; the sampling frequency
        is 1/dt Hz.
    b: float
        Encoder bias.
    d: float
        Encoder threshold.
    k: float
        Encoder integration constant.
    dte: float
        Sampling resolution assumed by the encoder (s).
        This may not exceed dt.

    """

    # Use a generator to split the input array into chunks:
    ### NOTE: the blocksize selection should be made more robust than the
    ### implementation below.
    block_factor = 10
    length_limit = 1000
    block_size = len(u)/block_factor if len(u) > length_limit else len(u)
    g = misc.chunks(u, block_size)
    def get():
        try:
            temp = g.next()
        except StopIteration:
            return array([], float)
        else:
            return array(temp)
    s = []
    put = s.extend
    tem = RealTimeEncoder(get, put, dt, b, d, k, dte)
    tem.run()
    return array(s)

def asdm_decode_real(s, dur, dt, bw, b, d, k=1.0, N=10, M=2, K=1):    
    """Decode an arbitrarily long signal encoded by an asynchronous
    sigma-delta modulator.

    Parameters
    ----------
    s: numpy array of floats
        Encoded signal. The values represent the time between spikes (in s).
    dur: float
        Duration of signal (in s).
    dt: float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw: float
        Signal bandwidth (in rad/s).
    b: float
        Encoder bias.
    d: float
        Encoder threshold.
    k: float
        Encoder integrator constant.
    N: int
        Number of spikes to process in each block less 1.
    M: int
        Number of spikes between the starting time of each successive
        block.
    K: int
        Number of spikes in the overlap between successive blocks.   
        
    """

    # Use a generator to split the input array into chunks:
    ### NOTE: the blocksize selection should be made more robust than the
    ### implementation below.
    block_factor = 10
    length_limit = 1000
    block_size = len(s)/block_factor if len(s) > length_limit else len(s)
    g = misc.chunks(s, block_size)
    def get():
        try:
            temp = g.next()
        except StopIteration:
            return array([], float)
        else:
            return array(temp)    
    u = []
    put = u.extend
    tdm = RealTimeDecoder(get, put, dt, bw, b, d, k, N, M, K)
    tdm.run()
    return array(u)

def asdm_decode_ins_real(s, dur, dt, bw, b, N=10, M=2, K=1):    
    """Decode an arbitrarily long signal encoded by an asynchronous
    sigma-delta modulator using a threshold-insensitive recovery
    algorithm.

    Parameters
    ----------
    s: numpy array of floats
        Encoded signal. The values represent the time between spikes (in s).
    dur: float
        Duration of signal (in s).
    dt: float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw: float
        Signal bandwidth (in rad/s).
    b: float
        Encoder bias.
    N: int
        Number of spikes to process in each block less 1.
    M: int
        Number of spikes between the starting time of each successive
        block.
    K: int
        Number of spikes in the overlap between successive blocks.   
        
    """

    # Use a generator to split the input array into chunks:
    ### NOTE: the blocksize selection should be made more robust than the
    ### implementation below.
    block_factor = 10
    length_limit = 1000
    block_size = len(s)/block_factor if len(s) > length_limit else len(s)
    g = misc.chunks(s, block_size)
    def get():
        try:
            temp = g.next()
        except StopIteration:
            return array([], float)
        else:
            return array(temp)    
    u = []
    put = u.extend
    tdm = RealTimeDecoderIns(get, put, dt, bw, b, N, M, K)
    tdm.run()
    return array(u)
