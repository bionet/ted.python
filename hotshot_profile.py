#!/usr/bin/env python

"""
Profile a program using hotshot.
"""

import hotshot
import getopt
import sys
import os

def usage():
    print "Usage: %s [-o outfile] script" % sys.argv[0]
    print "Profile a Python program using hotshot."
    sys.exit(0)
    
(opts,args) = getopt.getopt(sys.argv[1:],'ho:')

outfile = 'profile.out'
for o in opts:
    if o[0] == '-h':
        usage()
        sys.exit(0)
    elif o[0] == '-o':
        outfile = o[1]
    else:
        pass

if not len(args):
    usage()
infile = args[0]
if not os.path.exists(infile):
    print '%s does not exist' % infile
    sys.exit(1)

p = hotshot.Profile(outfile,lineevents=0)
p.run("execfile('%s')" % infile)
p.close()



