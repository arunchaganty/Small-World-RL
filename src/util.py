"""
RL Framework
Authors: Arun Chaganty
General utility functions
"""

import sys
import numpy as np
from ProgressBar import ProgressBar

def normalise( dist ):
    return dist / np.sum( dist )
    
def choose( dist ):
    vs, dist = zip( *dist )
    dist = normalise( dist )
    idx = np.random.multinomial( 1, dist ).argmax()
    return vs[ idx ]

def progressIter( fn, lst ):

    progress = ProgressBar( 0, len(lst), mode='fixed' )
    oldprog = str(progress)

    for n in lst:
        v = fn( n )

        if v:
            progress.update_amount(v)
        else:
            progress.increment_amount()
        if oldprog != str(progress):
            print progress, "\r",
            sys.stdout.flush()
            oldprog=str(progress)
    print '\n'

def progressMap( fn, lst ):

    progress = ProgressBar( 0, len(lst), mode='fixed' )
    oldprog = str(progress)

    out = []

    for n in lst:
        v = fn( n )
        out.append( v )

        progress.increment_amount()
        if oldprog != str(progress):
            print progress, "\r",
            sys.stdout.flush()
            oldprog=str(progress)
    print '\n'

    return out

