#!/usr/bin/env python

"""
Capture a video from a webcam using OpenCV.
"""

import time
import cv
import video_io as v

camera = cv.CaptureFromCAM(0)
filename = 'test.avi'
fourcc = ('D','I','V','X')
w = v.WriteVideo(filename, fourcc, 30.0, (640,480), True)

start = time.time()
end = start + 15
while time.time() < end:
    frame = cv.QueryFrame(camera)
    w.write_cv_frame(frame)
    
