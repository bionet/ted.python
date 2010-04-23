import numpy as np
import cv

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
    
    def read_np_frame(self):
        """Read a frame as a numpy array from the video."""
        
        frame = cv.QueryFrame(self.capture)
        if frame != None:
            return self.__cv2array(frame)
        else:
            return None

    def read_cv_frame(self):
        """Read a frame as an OpenCV image from the video."""
        
        frame = cv.QueryFrame(self.capture)
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
