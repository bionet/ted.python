#!/usr/bin/env python

"""
Video I/O classes
=================

Classes for reading and writing video files into and from numpy [1]
ndarrays using OpenCV [2]_ and
matplotlib [3]_.

- ReadVideo          - Read frames from a video into ndarrays.
- WriteVideo         - Write ndarrays as frames of a video.
- WriteFigureVideo   - Write matplotlib figures as frames of a video.
- video_capture      - Capture video data from a webcam.

.. [1] http://numpy.scipy.org/
.. [2] http://opencv.willowgarage.com/wiki/PythonInterface/
.. [3] http://matplotlib.sf.net/

"""

__all__ = ['ReadVideo', 'WriteVideo', 'WriteFigureVideo',
           'video_capture']

import numpy as np
import cv

from numpy import floor
import matplotlib

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import subprocess
import os
import tempfile

from time import time
from glob import glob

if not os.path.exists('/usr/bin/mencoder'):
    raise RuntimeError('mencoder not found')

# See http://opencv.willowgarage.com/wiki/PythonInterface
# for more information on converting OpenCV image to ndarrays and
# vice versa.

class ReadVideo:
    """
    This class provides an interface for reading frames from a video
    file using OpenCV and returning them as ndarrays.

    Parameters
    ----------
    filename : string
        Name of input video file.

    Methods
    -------
    get_frame_count()
        Return the number of frames in the video.
    get_prop_fps()
        Return the frame rate of the video.
    read_cv_frame()
        Read a frame from the video as an OpenCV frame.
    read_np_frame()
        Read a frame from the video as an ndarray.
        
    """

    def __init__(self, filename):
        self.capture = cv.CaptureFromFile(filename)

    def get_frame_count(self):
        """Return the number of frames in the file."""

        return int(cv.GetCaptureProperty(self.capture,
                                         cv.CV_CAP_PROP_FRAME_COUNT))

    def get_prop_fps(self):
        """Return the frame rate of the video file."""

        return cv.GetCaptureProperty(self.capture,
                                     cv.CV_CAP_PROP_FPS)
    
    def __cv2array(self, img):
        """Convert an OpenCV image to an ndarray of dimensions
        (height, width, channels)."""
        
        depth2dtype = {
            cv.IPL_DEPTH_8U: 'uint8',
            cv.IPL_DEPTH_8S: 'int8',
            cv.IPL_DEPTH_16U: 'uint16',
            cv.IPL_DEPTH_16S: 'int16',
            cv.IPL_DEPTH_32S: 'int32',
            cv.IPL_DEPTH_32F: 'float32',
            cv.IPL_DEPTH_64F: 'float64',
            }

        arrdtype = img.depth
        a = np.fromstring(img.tostring(),
                          dtype=depth2dtype[img.depth],
                          count=img.width*img.height*img.nChannels)

        # numpy assumes that the first dimension of an image is its
        # height, i.e., the number of rows in the image:
        a.shape = (img.height, img.width, img.nChannels)
        return a

    def __get_frame(self, n):
        """Retrieve the specified frame from the video."""

        if n != None:
            cv.SetCaptureProperty(self.capture,
                                  cv.CV_CAP_PROP_POS_FRAMES, n)
        return cv.QueryFrame(self.capture)
    
    def read_np_frame(self, n=None):
        """Read a frame as a numpy array from the video."""
        
        frame = self.__get_frame(n)
        if frame != None:
            return self.__cv2array(frame)
        else:
            return None

    def read_cv_frame(self, n=None):
        """Read a frame as an OpenCV image from the video."""
        
        frame = self.__get_frame(n)
        return frame

class WriteVideo:
    """
    This class provides an interface for writing frames represented
    as ndarrays to a video file using OpenCV.

    Parameters
    ----------
    filename : string
        Name of output video file.
    fourcc : tuple
        Video codec of output video; default is ('M', 'J', 'P', 'G').
    fps : float
        Frame rate of output video; default is 30.0.
    frame_size : tuple 
        Size of video frames (rows, columns); default is (256, 256).
    is_color : bool
        True of the video is color.

    Methods
    -------
    write_cv_frame(a)
        Write an OpenCV frame `a` to the video.
    write_np_frame(a)
        Write the frame represented as ndarray `a` to the video.
        
    """

    def __init__(self, filename, fourcc=('D', 'I', 'V', 'X'),
                 fps=30.0, frame_size=(256,256), is_color=True):
        self.writer = cv.CreateVideoWriter(filename,
                 cv.CV_FOURCC(*fourcc), fps, frame_size, int(is_color))


    def __array2cv(self, a):
        """Convert an ndarray to an OpenCV image."""
        
        dtype2depth = {
            'uint8':   cv.IPL_DEPTH_8U,
            'int8':    cv.IPL_DEPTH_8S,
            'uint16':  cv.IPL_DEPTH_16U,
            'int16':   cv.IPL_DEPTH_16S,
            'int32':   cv.IPL_DEPTH_32S,
            'float32': cv.IPL_DEPTH_32F,
            'float64': cv.IPL_DEPTH_64F,
            }
        try:
            nChannels = a.shape[2]
        except:
            nChannels = 1

        # numpy assumes that the first dimension of an image is its
        # height, i.e., the number of rows in the image:
        img = cv.CreateImageHeader((a.shape[1], a.shape[0]),
                                   dtype2depth[str(a.dtype)],
                                   nChannels)
        cv.SetData(img, a.tostring(),
                   a.dtype.itemsize*nChannels*a.shape[1])
        return img
          
    def write_cv_frame(self, a):
        """Write an OpenCV image as a frame to the video."""
        
        cv.WriteFrame(self.writer, a)
        
    def write_np_frame(self, a):
        """Write a numpy array as a frame to the video."""
        
        img = self.__array2cv(a)
        cv.WriteFrame(self.writer, img)

class WriteFigureVideo:
    """
    This class provides an interface for writing frames represented as
    matplotlib figures to a video file using mencoder.

    Parameters
    ----------
    filename : str
        Output video file name.
    dpi : float
        Resolution at which to save each frame. This defaults to
        the value assumed by `savefig`.
    width : float
        Frame width (in inches).
    height : float
        Frame height (in inches).
    fps : float
        Frames per second.

    Methods
    -------
    write_fig(fig)
        Write a matplotlib figure `fig` to the output video file.
    create_video()
        Create the output video file.
        
    Notes
    -----
    This class is based upon the file movie_demo.py ((c) 2004 by Josh
    Lifton) included with matplotlib. The output video file is not
    actually assembled until the `close` method is called.
    
    """

    def __init__(self, filename,
                 dpi=matplotlib.rcParams['savefig.dpi'], width=8.0,
                 height=6.0, fps=25):
        self.filename = filename
        self.dpi = dpi
        self.width = 8.0
        self.height = 6.0
        self.fps = fps
        
        self.tempdir = tempfile.mkdtemp()
        self.frame_count = 0
        
    def write_fig(self, fig):
        """Write a matplotlib figure to the output video file."""
        
        if self.tempdir == None:
            raise ValueError('cannot add frames to completed video file')
        
        if not isinstance(fig, Figure):
            raise ValueError('can only write instances of type '
                             'matplotlib.figure.Figure')
        if fig.get_figwidth() != self.width:
            raise ValueError('figure width must be %f' % self.width)
        if fig.get_figheight() != self.height:
            raise ValueError('figure height must be %f' % self.height)
        
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(self.tempdir + str("/%010d.png" % self.frame_count),
                            self.dpi)
        self.frame_count += 1
        
    def create_video(self):
        """Assemble the output video from the input frames."""
        
        width_pix = int(floor(self.width*self.dpi))
        height_pix = int(floor(self.height*self.dpi))
        command = ('mencoder', 'mf://'+self.tempdir+'/*.png',
                   '-mf', 'type=png:w=%d:h=%d:fps=%d' % (width_pix, height_pix,
                                                         self.fps), 
                   '-ovc', 'lavc', '-lavcopts', 'vcodec=mpeg4',
                   '-oac', 'copy', '-o', self.filename)
        subprocess.check_call(command)
        for f in glob(self.tempdir + '/*.png'):
            os.remove(f)
        os.rmdir(self.tempdir)
        self.tempdir = None
        
    def __del__(self):
        """Create the video before the class instance is destroyed."""
        
        if self.tempdir:
            self.create_video()
    

def video_capture(filename, t, fourcc=('D','I','V','X'), fps=30.0,
                  frame_size=(640, 480), is_color=True):
    """Capture a video of time length `t` from a webcam using OpenCV
    and save it in `filename` using the specified format."""

    camera = cv.CaptureFromCAM(0)
    w = WriteVideo(filename, fourcc, fps, frame_size, is_color)
    start = time()
    end = start + t
    while time() < end:
        frame = cv.QueryFrame(camera)
        w.write_cv_frame(frame)
