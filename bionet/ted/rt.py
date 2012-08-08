#!/usr/bin/env python

"""
Real-time time encoding and decoding algorithms. These algorithms
can process signals of arbitrarily long length without the memory
limitations of the functions in the asdm and iaf modules.

- ASDMRealTimeDecoder    - Real-time ASDM decoder.
- ASDMRealTimeDecoderIns - Parameter-insensitive real-time ASDM decoder.
- ASDMRealTimeEncoder    - Real-time ASDM encoder.
- IAFRealTimeEncoder     - Real-time IAF encoder.
- IAFRealTimeDecoder     - Real-time IAF decoder.
- iaf_decode_delay_rt    - Real-time delayed IAF decoder.
- iaf_encode_delay_rt    - Real-time delayed IAF encoder.

"""

__all__ = ['SignalProcessor',
           'RealTimeEncoder', 'RealTimeDecoder',
           'ASDMRealTimeEncoder', 'ASDMRealTimeDecoder',
           'ASDMRealTimeDecoderIns',
           'IAFRealTimeEncoder', 'IAFRealTimeDecoder',
           'iaf_decode_delay_rt', 'iaf_encode_delay_rt']

# Setting this flag enables the silent generation of a debug plot
# depicting the progress of the stitching algorithm employed in the
# real-time decoders.
debug = False
if debug:
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        debug = False
    else:
        debug_plot_filename = 'rt_debug.png'
        debug_plot_figsize = (7, 5)
        debug_plot_dpi = 100

import numpy as np

import bionet.utils.misc as m
import bionet.utils.numpy_extras as ne
import bionet.ted.asdm as asdm
import bionet.ted.iaf as iaf
import bionet.ted.vtdm as vtdm

class SignalProcessor(object):
    """
    Abstract signal processor.

    This class describes a signal processor that retrieves blocks
    of signal data from a source, applies some processing algorithm to
    it, and saves the processed blocks.

    Methods
    -------
    process(get, put)
        Process data obtained from `get()` and write it using `put()`.

    Notes
    -----
    The `process()` method must be extended in functional subclasses of this
    class in order.

    """

    def __init__(self, *args):
        """Initialize a signal processor with the specified
        parameters."""

        self.params = args

    def __call__(self, x):
        """Calling a class instance is equivalent to running the
        processor on the specified sequence `x`."""

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
        """Process data obtained in blocks from the function `get()`
        and write them out using the function `put()`."""

        if not callable(get):
            raise ValueError('get() must be callable')
        if not callable(put):
            raise ValueError('put() must be callable')

    def __repr__(self):
        """Represent a signal processor in terms its parameters."""

        return self.__class__.__name__+repr(tuple(self.params))

class RealTimeEncoder(SignalProcessor):
    """
    Abstract real-time time encoding machine.

    This class implements a real-time time encoding machine. It
    must be subclassed to use a specific encoding algorithm.

    Methods
    -------
    encode(data, ...)
        Encode a block of data using the additional parameters.
    process(get, put)
        Process data obtained from `get()` and write it using `put()`.

    Notes
    -----
    The `encode()` method must be extended to contain a time encoding
    algorithm implementation in functional subclasses of this class.

    """

    def __init__(self, *args):
        """Initialize a real-time time encoder."""

        SignalProcessor.__init__(self, *args)

    def encode(self, *args):
        """Encode a block of data. This method must be reimplemented
        to use a specific encoding algorithm implementation."""

        pass

    def process(self, get, put):
        """Encode data returned in blocks by function `get()` and
        write it to some destination using the function `put()`."""

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

class RealTimeDecoder(SignalProcessor):
    """
    Abstract real-time time decoding machine.

    This class implements a real-time time decoding machine. It
    must be subclassed to use a specific decoding algorithm.

    Parameters
    ----------
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is `1/dt` Hz.
    bw : float
        Signal bandwidth (in rad/s).
    N : int
        Number of spikes to process in each block less 1.
    M : int
        Number of spikes between the starting time of each successive
        block.
    K : int
        Number of spikes in the overlap between successive blocks.

    Methods
    -------
    decode(data, ...)
        Decode a block of data using the additional parameters.
    process(get, put)
        Process data obtained from `get()` and write it using `put()`.

    """

    def __init__(self, dt, bw, N, M, K):

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
        self.sgn = 1

        # Spike intervals and spike indicies:
        self.s = []
        self.tk = np.array((), np.float)

        # Overlap:
        self.overlap = np.array((), np.float)

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

        if debug:
            self.fig = Figure(figsize=debug_plot_figsize)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel('t (s)')
            self.ax.set_ylabel('y(t)')
            self.offset = 0.0

    def decode(self, *args):
        """Decode a block of data. This method must be reimplemented
        to use a specific decoding algorithm implementation."""

        pass

    def process(self, get, put):
        """Decode data returned in blocks by function `get()` and
        write it to some destination using the function `put()`."""

        SignalProcessor.process(self, get, put)

        # Set up a buffer to queue input data from the source:
        # XXX: the number of initial entries here is arbitrary:
        sb = m.SerialBuffer(get, 10*self.N)

        while True:

            # Get new data to add to the block of encoded data to be
            # decoded:
            self.intervals_to_add = sb.read(self.intervals_needed)

            # If the number of intervals actually obtained is less
            # than that requested, then the final block has been
            # reached and hence should not be windowed on its right side:
            if len(self.intervals_to_add) < self.intervals_needed:
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
            self.ts = np.cumsum(self.s)
            self.tk = np.array(np.round(self.ts/self.dt), int)
            self.curr_dur = max(self.ts)
            self.t = np.arange(0, self.curr_dur, self.dt)

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
                self.sgn *= -1

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

            if debug:
                self.ax.plot(self.offset+self.t, self.uw)

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
                self.u_out = np.hstack((self.u_out,
                    self.uw[self.tk[self.M+self.K]:self.tk[self.N-self.M-self.K]]))
                self.overlap = \
                    self.uw[self.tk[self.N-self.M-self.K]:self.tk[self.N-self.M]]
                if debug:
                    self.offset += self.t[self.tk[self.J-1]]
            else:
                self.u_out = np.hstack((self.u_out,
                                     self.uw[self.tk[self.M+self.K]::]))
                self.overlap = np.array((), np.float)
                if debug:
                    self.offset += 0

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

    def __call__(self, x):
        """Calling a class instance is equivalent to running the
        decoder on the specified sequence `x`."""

        result = SignalProcessor.__call__(self, x)
        if debug:
            self.canvas = FigureCanvasAgg(self.fig)
            self.canvas.print_figure(debug_plot_filename,
                                     debug_plot_dpi)
        return result

    # Methods for computing the edges of the windows determined by the
    # windows() method:
    def _theta1(self, t, l, r):
        return np.sin((np.pi/2)*(t-l)/(r-l))**2

    def _theta2(self, t, l, r):
        return np.cos((np.pi/2)*(t-l)/(r-l))**2

    def window(self, t, ll, lr, rl, rr):
        """Return a window defined over the vector of times t that
        forms a partition of unity over all time. The function is
        equal to 0 when t <= ll or t > rr, theta(t,ll,lr) when ll < t
        <= lr, 1 when lr < t <= rl, and 1-theta(t,rl,rr)
        when rl < t <= rr."""

        w = np.zeros(len(t), np.float)

        i1 = np.intersect1d(np.where(ll < t)[0], np.where(t <= lr)[0])
        i2 = np.intersect1d(np.where(lr < t)[0], np.where(t <= rl)[0])
        i3 = np.intersect1d(np.where(rl < t)[0], np.where(t <= rr)[0])

        w[i1] = self._theta1(t[i1], ll, lr)
        w[i2] = 1.0
        w[i3] = self._theta2(t[i3], rl, rr)

        return w

class ASDMRealTimeEncoder(RealTimeEncoder):
    """
    Real-time ASDM time encoding machine.

    This class implements a real-time time encoding machine that uses
    an Asynchronous Sigma-Delta Modulator to encode data.

    Parameters
    ----------
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is `1/dt` Hz.
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    k : float
        Encoder integration constant.
    dte : float
        Sampling resolution assumed by the encoder. This may not exceed
        `dt`.
    quad_method : {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal).


    Methods
    -------
    encode(data, ...)
        Encode a block of data using the additional parameters.
    process(get, put)
        Process data obtained from `get()` and write it using `put()`.

    """

    def __init__(self, dt, b, d, k=1.0, dte=0.0, quad_method='trapz'):

        # The values 0, 0, and 1 passed to the constructor
        # initialize the y, interval, and sgn parameters of the ASDM
        # encoder function:
        SignalProcessor.__init__(self, dt, b, d, k, dte, 
                                 0.0, 0.0, 1, quad_method, True)

    def encode(self, data):
        """Encode a block of data with an ASDM encoder."""

        return asdm.asdm_encode(data, *self.params)

class ASDMRealTimeDecoder(RealTimeDecoder):
    """
    Real-time ASDM time decoding machine.

    This class implements a real-time time decoding machine that
    decodes data encoded using an Asynchronous Sigma-Delta Modulator.

    Parameters
    ----------
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is 1/dt Hz.
    bw : float
        Signal bandwidth (in rad/s).
    b : float
        Encoder bias.
    d : float
        Decoder threshold.
    k : float
        Decoder integration constant.
    N : int
        Number of spikes to process in each block less 1.
    M : int
        Number of spikes between the starting time of each successive
        block.
    K : int
        Number of spikes in the overlap between successive blocks.

    Methods
    -------
    decode(data, ...)
        Decode a block of data using the additional parameters.
    process(get, put)
        Process data obtained from `get()` and write it using `put()`.

    """

    def __init__(self, dt, bw, b, d, k, N, M, K):

        RealTimeDecoder.__init__(self, dt, bw, N, M, K)

        self.b = b
        self.d = d
        self.k = k

    def decode(self, data):
        """Decode a block of data that was encoded with an ASDM
        encoder."""

        return vtdm.asdm_decode_vander(data, self.curr_dur, self.dt,
                                       self.bw, self.b, self.d, self.k,
                                       self.sgn)

class ASDMRealTimeDecoderIns(RealTimeDecoder):
    """
    Real-time threshold-insensitive ASDM time decoding machine.

    This class implements a threshold-insensitive real-time time
    decoding machine that decodes data encoded using an Asynchronous
    Sigma-Delta Modulator.

    Parameters
    ----------
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is `1/dt` Hz.
    bw : float
        Signal bandwidth (in rad/s).
    b : float
        Encoder bias.
    N : int
        Number of spikes to process in each block less 1.
    M : int
        Number of spikes between the starting time of each successive
        block.
    K : int
        Number of spikes in the overlap between successive blocks.

    Methods
    -------
    decode(data, ...)
        Decode a block of data using the additional parameters.
    process(get, put)
        Process data obtained from `get()` and write it using `put()`.

    """

    def __init__(self, dt, bw, b, N, M, K):

        RealTimeDecoder.__init__(self, dt, bw, N, M, K)

        self.b = b

    def decode(self, data):
        """Decode a block of data that was encoded with an ASDM
        encoder."""

        return vtdm.asdm_decode_vander_ins(data, self.curr_dur, self.dt,
                                           self.bw, self.b, self.sgn)

class IAFRealTimeEncoder(RealTimeEncoder):
    """
    Real-time IAF neuron time encoding machine.

    This class implements a real-time time encoding machine that uses
    an Integrate-and-Fire neuron to encode data.

    Parameters
    ----------
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is `1/dt` Hz.
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.
    dte : float
        Sampling resolution assumed by the encoder. This may not exceed
        `dt`.
    quad_method : {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal).

    Methods
    -------
    encode(data, ...)
        Encode a block of data using the additional parameters.
    process(get, put)
        Process data obtained from `get()` and write it using `put()`.

    """

    def __init__(self, dt, b, d, R=np.inf, C=1.0, dte=0.0, quad_method='trapz'):

        # The values 0 and 0 passed to the constructor initialize the
        # y and interval parameters of the IAF encoder function:
        SignalProcessor.__init__(self, dt, b, d, R, C, dte,
                                 0.0, 0.0, quad_method, True)

    def encode(self, data):
        """Encode a block of data with an IAF neuron."""

        return iaf.iaf_encode(data, *self.params)

class IAFRealTimeDecoder(RealTimeDecoder):
    """
    Real-time IAF neuron time decoding machine.

    This class implements a real-time time decoding machine that
    decodes data encoded using an Integrate-and-Fire neuron.

    Parameters
    ----------
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is `1/dt` Hz.
    bw : float
        Signal bandwidth (in rad/s).
    b : float
        Encoder bias.
    d : float
        Decoder threshold.
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.
    N : int
        Number of spikes to process in each block less 1.
    M : int
        Number of spikes between the starting time of each successive
        block.
    K : int
        Number of spikes in the overlap between successive blocks.

    Methods
    -------
    decode(data, ...)
        Decode a block of data using the additional parameters.
    process(get, put)
        Process data obtained from `get()` and write it using `put()`.

    """

    def __init__(self, dt, bw, b, d, R, C, N, M, K):

        RealTimeDecoder.__init__(self, dt, bw, N, M, K)

        self.b = b
        self.d = d
        self.R = R
        self.C = C

    def decode(self, data):
        """Decode a block of data that was encoded with an
        IAF neuron."""

        return vtdm.iaf_decode_vander(data, self.curr_dur, self.dt,
                                      self.bw, self.b, self.d, self.R, self.C)

def iaf_encode_delay_rt(u_list, T_block, t_begin, dt,
                        b_list, d_list, k_list, a_list, w_list):
    """
    Real-time multi-input multi-output delayed IAF time encoding machine.

    Encode several with an ensemble of ideal Integrate-and-Fire
    neurons with delays.

    Parameters
    ----------
    u_list : list
        Signals to encode. Must contain `M` arrays of equal length.
    T_block : float
        Length of block to encode (in s) during each iteration.
    t_begin : float
        Time at which to begin encoding (in s).
    dt : float
        Sampling resolution of input signals; the sampling frequency
        is 1/dt Hz.
    b_list : list
        List of encoder biases. Must be of length `M`.
    d_list : list
        List of encoder thresholds. Must be of length `M`.
    k_list : list
        List of encoder integration constants. Must be of length `M`.
    a_list : array_like
        Array of neuron delays (in s). Must have shape `(N, M)`.
    w_list : array_like
        Array of scaling factors. Must have shape `(N, M)`.

    Returns
    -------
    s_list : list
        List of arrays of interspike intervals.

    """

    M = len(u_list)
    if not M:
        raise ValueError('no spike data given')

    if len(set(map(len, u_list))) > 1:
        raise ValueError('all input signals must be of the same length')
    Nt = len(u_list[0])

    N = len(b_list) # number of neurons

    # Initialize interspike interval storage lists:
    s_list = [[] for i in xrange(N)]

    # Initialize integrator and current interspike interval arrays:
    interval_list = [0.0 for i in xrange(N)]
    y_list = [0.0 for i in xrange(N)]

    # Convert times to integer indicies to avoid index round-off problems:
    if T_block <= t_begin:
        raise ValueError('block length must exceed start time')

    a_max = np.max(a_list)
    T = T_block
    t_start = 0.0
    t_end = T

    K = ne.ifloor((T-t_begin)/dt)
    k_start = 0
    k_end = ne.ifloor(t_end/dt)

    count = 0
    while k_start < Nt:

        # Convert the bounds of the interval to encode to indices:
        print '%i: window: [%f, %f]' % (count, k_start*dt, k_end*dt)
        count += 1

        # Encode the block:
        u_block_list = map(lambda x: x[k_start:k_end], u_list)
        s_curr_list, t_begin, dt, b_list, d_list, k_list, a_list, \
                     w_list, y_list, interval_list, full_output = \
                     iaf.iaf_encode_delay(u_block_list,
                                          t_begin,
                                          dt, b_list, d_list, k_list,
                                          a_list, w_list,
                                          y_list, interval_list, True)

        # Save the encoded data:
        for i in xrange(N):
            s_list[i].extend(s_curr_list[i])

        # Advance k_start and k_end:
        k_start += K
        k_end += K

        # When the end of the signal is reached, the encoding block
        # must be shortened:
        if k_end > Nt:
            k_end = Nt

    return map(np.asarray, s_list)

def _theta1(t, l, r):
    return np.sin((np.pi/2)*(t-l)/(r-l))**2

def _theta2(t, l, r):
    return np.cos((np.pi/2)*(t-l)/(r-l))**2

def _get_spike_block(s_list, t_start, t_end):
    """
    Get block of interspike intervals.

    If `s_list` contains arrays of interspike intervals, return a list
    of subarrays containing those interspike intervals between the
    times `t_start` and `t_end`.

    Parameters
    ----------
    s_list : list
        List of interspike interval arrays.
    t_start : float
        Starting time of block.
    t_end : float
        Ending time of block.

    Returns
    -------
    s_block_list : list
        List of interspike interval subarrays containing the desired values.

    Notes
    -----
    The first interspike interval in each subarray in the returned
    block is adjusted to avoid introducing incorrect shifts between
    each spike train.

    """

    if t_end <= t_start:
        raise ValueError('t_end must exceed t_start')

    ts_list = map(np.cumsum, s_list)
    s_block_list = []
    for i in xrange(len(s_list)):
        block_indices = \
                      np.intersect1d(np.where(ts_list[i] > t_start)[0],
                                     np.where(ts_list[i] <= t_end)[0])
        s_block = s_list[i][block_indices].copy()

        # Adjust first interspike interval in the block:
        s_block[0] = ts_list[i][block_indices[0]]-t_start

        s_block_list.append(s_block.copy())

    return s_block_list

def iaf_decode_delay_rt(s_list, T_block, T_overlap, dt,
                        b_list, d_list, k_list, a_list, w_list):
    """
    Real-time multi-input multi-output delayed IAF time decoding machine.

    Decode several signals encoded with an ensemble of ideal
    Integrate-and-Fire neurons with delays.

    Parameters
    ----------
    s : list of ndarrays of floats
        Signals encoded by an ensemble of encoders. The values
        represent the time between spikes (in s). The number of arrays
        in the list corresponds to the number of encoders in the ensemble.
    T_block : float
        Length of block to decode during each iteration (in s).
    T_overlap : float
        Length of overlap between successive blocks (in s).
    dt : float
        Sampling resolution of input signals; the sampling frequency
        is 1/dt Hz.
    b_list : list
        List of encoder biases. Must be of length `N`.
    d_list : list
        List of encoder thresholds. Must be of length `N`.
    k_list : list
        List of encoder integration constants. Must be of length `N`.
    a_list : array_like
        Array of neuron delays (in s). Must be of shape `(N, M)`.
    w_list : array_like
        Array of scaling factors. Must be of shape `(N, M)`.

    Returns
    -------
    u_list : list
        Decoded signals.

    """

    if 2*T_overlap >= T_block:
        raise ValueError('overlap cannot exceed half of the block length')

    # Stitching the first and last blocks requires special treatment:
    first_block = True
    last_block = False

    # Convert times to integer indicies to avoid index round-off problems:
    K_block = ne.iround(T_block/dt)
    K = K_block
    K_overlap = ne.iround(T_overlap/dt)

    # How much to shift the decoding window during each iteration:
    K_inc = ne.iround((T_block-T_overlap)/dt)

    # Decoding window bounds:
    k_start = 0
    k_end = K

    # The portion of the last block that overlaps with the current
    # decoded block is stored here:
    u_overlap = None

    # The decoded blocks are accumulated in a list and concatenated
    # when the end of the signal is reached:
    u_block_list = []
    t_max = np.max(map(np.sum, s_list))
    k_max = ne.iround(t_max/dt)

    # Don't bother stitching if the encoded signal spans an interval
    # of time that is shorter than the block length:
    if k_max < K_block:
        return iaf.iaf_decode_delay(s_list, K*dt, dt, b_list, d_list,
                                          k_list, a_list, w_list)
    count = 0
    while k_start < k_max:

        # Select block of spike times to decode:
        s_block_list = _get_spike_block(s_list, k_start*dt, k_end*dt)
        print '%i: window: [%f, %f]' % (count, k_start*dt, k_end*dt)
        count += 1

        # Decode the block:
        u_curr_list = iaf.iaf_decode_delay(s_block_list, K*dt, dt, b_list, d_list,
                                                 k_list, a_list, w_list)

        # Convert decoded block into a 2D array to make processing easier:
        u_curr = np.array(u_curr_list)

        # The first block doesn't need to be stitched on its left side:
        if first_block:
            u_block_list.append(u_curr[:, 0:K_overlap])
        else:

            # Generate windowing functions needed to taper the overlap
            # from the previous iteration and the overlap from the
            # current iteration:
            win_prev = _theta2(np.arange(K_overlap, dtype=np.float), 0,
                               K_overlap)
            win_curr = _theta1(np.arange(K_overlap, dtype=np.float), 0,
                               K_overlap)

            # Stitch and save the overlapping portion of the block:
            u_block_list.append(u_overlap*win_prev+\
                                u_curr[:, 0:K_overlap]*win_curr)

        if last_block:

            # Save the rest of the current block and exit:
            u_block_list.append(u_curr[:, K_overlap:])
            break
        else:

            # Save the portion of the block that doesn't require stitching:
            u_block_list.append(u_curr[:, K_overlap:-K_overlap])

            # Retain the overlap on the right side of the decoded
            # block for the next iteration:
            u_overlap = u_curr[:, -K_overlap:]

        # Advance t_start and t_end allowing for an overlap:
        k_start += K_inc
        k_end += K_inc

        # When the end of the signal is reached, the decoding block
        # must be shortened:
        if k_end >= k_max:
            k_end = k_max
            K = k_end-k_start
            last_block = True
        else:
            K = K_block

        # Indicate that the first block has been processed:
        if first_block:
            first_block = False
            
    # Concatenate all of the decoded blocks and return as a list of arrays:
    return list(np.hstack(u_block_list))
