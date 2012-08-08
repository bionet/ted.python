#!/usr/bin/env python

"""
Convert all PGF diagrams in the images/ subdirectory to images.
"""

from glob import glob
import os

from pgf2img import pgf2img

output_ext = '.png'
os.chdir('images')
pgf_files = glob('*.pgf')
for input_filename in pgf_files:
    output_filename = os.path.splitext(input_filename)[0] + output_ext
    print 'converting %s to %s' % (input_filename, output_filename)
    pgf2img(input_filename, output_filename)

