#!/usr/bin/env python

"""
Real-time time encoding and decoding algorithms.
"""

__all__ = ['SignalProcessor',
           'RealTimeEncoder', 'RealTimeDecoder',
           'ASDMRealTimeEncoder', 'ASDMRealTimeDecoder',
           'ASDMRealTimeDecoderIns']

import bionet.utils.misc as m
import bionet.ted.asdm as asdm
import bionet.ted.vtdm as vtdm

from numpy import arange, array, cos, cumsum, float, hstack, \
     intersect1d, pi, round, sin, where, zeros

class SignalProcessor:
    """This class describes a signal processor that retrieves blocks
    of signal data from a source, applies some processing algorithm to
    it, and saves the processed blocks. It must be subclassed in order
    to be made functional."""

    def __init__(self, *args):
        """Initialize a signal processor with the specified
        parameters."""

        self.params = args

    def __call__(self, x):
        """Calling a class instance is equivalent to running the
        processor on the specified sequence x."""

        result = []
        iterator = m.chunks(x, len(x)/10)
        def get():
            try:
                return iterator.next()
            except StopIteration:
                return []
        def put(y):
            result.extend(y)
            
        self.process(get, put)
        return result
    
    def process(self, get, put):
        """Process data obtained in blocks from the function get()
        and write them out using the function put()."""

        if not callable(get):
            raise ValueError('get() must be callable')
        if not callable(put):
            raise ValueError('put() must be callable')

class RealTimeEncoder(SignalProcessor):
    """This class implements a real-time time encoding machine. It
    must be subclassed to use a specific encoding algorithm."""

    def __init__(self, *args):
        """Initialize a real-time time encoder."""

        SignalProcessor.__init__(self, *args)

    def encode(self, *args):
        """Encode a block of data. This method must be reimplemented
        to use a specific encoding algorithm implementation."""

        pass
    
    def process(self, get, put):
        """Time encode data returned in blocks by function get() and
        write it to some destination using the function put()."""

        SignalProcessor.process(self, get, put)

        # The invocation of self.encode() assumes that the method
        # returns a tuple containing processed data in its first entry
        # followed by all of the parameters be passed back to the
        # method in subsequent invocations:
        while True:
            input_data = get()
            if len(input_data) == 0:
                break
            temp = self.encode(input_data)
            encoded_data = temp[0]
            self.params = temp[1:]
            put(encoded_data)

    def __repr__(self):
        return self.__class__.__name__+repr(tuple(self.params))
        
class ASDMRealTimeEncoder(RealTimeEncoder):
    """This class implements a real-time time encoding machine that
    uses an asynchronous sigma-delta modulator to encode data."""
    
    def __init__(self, dt, b, d, k=1.0, dte=0.0, quad_method='trapz'):
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
        quad_method: {'rect', 'trapz'}
            Quadrature method to use (rectangular or trapezoidal).
        """

        # The values 0, 0, and 1 passed to the constructor
        # initialize the y, interval, and sgn parameters of the ASDM
        # encoder function:
        SignalProcessor.__init__(self, dt, b, d, k, dte, 
                                 0.0, 0.0, 1, quad_method, True)
        
    def encode(self, data):
        """Encode a block of data using an asynchronous sigma-delta
        modulator.""" 
        
        return asdm.asdm_encode(data, *self.params)
    
class RealTimeDecoder(SignalProcessor):
    """This class implements a real-time time decoding machine."""
    
    def __init__(self, dt, bw, N, M, K):
        """Initialize a real-time time decoder.

        Parameters
        ----------
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

        SignalProcessor.__init__(self, dt, bw, N, M, K)

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

    def decode(self, *args):
        """Decode a block of data. This method must be reimplemented
        to use a specific decoding algorithm implementation."""

        pass

    def process(self, get, put):
        """Time decode data returned in blocks by function get() and
        write it to some destination using the function put()."""

        SignalProcessor.process(self, get, put)

        # Set up a buffer to queue input data from the source:
        # XXX: the number of initial entries here is arbitrary:
        sb = m.SerialBuffer(get, 10*self.N)

        while True:

            # Get new data to add to the block of encoded data to be
            # decoded:
            self.intervals_to_add = sb.read(self.intervals_needed)
                        
            # If no data was obtained, the final block has been
            # reached and hence should not be windowed on its right side:
            if len(self.intervals_to_add) == 0:
                self.window_right = False

            # Add the read data to the block to be decoded:
            self.s.extend(self.intervals_to_add)
            
            # After the first block, the number of extra spike
            # intervals to read during subsequent iterations should be
            # equal to J:
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
            self.u = self.decode(self.s)

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
                self.u_out = self.overlap + \
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
                self.u_out = hstack((self.u_out,
                                     self.uw[self.tk[self.M+self.K]::]))
                self.overlap = array((), float)
            
            # The first block decoded should only be windowed on its
            # right side if at all. Hence, if window_left is false and
            # window_right is true, window_left should be set to true
            # before the next block is processed so that it is
            # windowed on its left side:
            if self.window_right and not self.window_left:
                self.window_left = True
            
            # Write out the current decoded block:
            put(self.u_out)

            # If window_left is true and window_right is false, the
            # last block has been decoded and processing is complete:
            if not self.window_right:
                break

    # Methods for computing the edges of the windows:
    def _theta1(self, t, l, r):
        return sin((pi/2)*(t-l)/(r-l))**2

    def _theta2(self, t, l, r):
        return cos((pi/2)*(t-l)/(r-l))**2

    def window(self, t, ll, lr, rl, rr):
        """Return a window defined over the vector of times t that
        forms a partition of unity over all time. The function is
        equal to 0 when t <= ll or t > rr, theta(t,ll,lr) when ll < t
        <= lr, 1 when lr < t <= rl, and 1-theta(t,rl,rr)
        when rl < t <= rr."""

        w = zeros(len(t), float)
        
        i1 = intersect1d(where(ll < t)[0], where(t <= lr)[0])
        i2 = intersect1d(where(lr < t)[0], where(t <= rl)[0])
        i3 = intersect1d(where(rl < t)[0], where(t <= rr)[0])
        
        w[i1] = self._theta1(t[i1], ll, lr)
        w[i2] = 1.0
        w[i3] = self._theta2(t[i3], rl, rr)
        
        return w

class ASDMRealTimeDecoder(RealTimeDecoder):
    """This class implements a real-time time decoding machine that
    decodes data encoded using an asynchronous sigma-delta
    modulator."""

    def __init__(self, dt, bw, b, d, k, N, M, K):
        """Initialize a real-time time decoder.

        Parameters
        ----------
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

        RealTimeDecoder.__init__(self, dt, bw, N, M, K)

        self.b = b
        self.d = d
        self.k = k

    def decode(self, data):
        """Decode a block of data that was encoded using an
        asynchronous sigma-delta modulator."""
        
        return vtdm.vander_decode(data, self.curr_dur, self.dt,
                                  self.bw, self.b, self.d, self.k,
                                  self.first_spike)

class ASDMRealTimeDecoderIns(RealTimeDecoder):
    """This class implements a parameter-insensitive real-time time
    decoding machine that decodes data encoded using an asynchronous
    sigma-delta modulator."""

    def __init__(self, dt, bw, b, N, M, K):
        """Initialize a real-time time decoder.

        Parameters
        ----------
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

        RealTimeDecoder.__init__(self, dt, bw, N, M, K)

        self.b = b

    def decode(self, data):
        """Decode a block of data that was encoded using an
        asynchronous sigma-delta modulator."""
        
        return vtdm.vander_decode_ins(data, self.curr_dur, self.dt,
                                      self.bw, self.b, self.first_spike)
