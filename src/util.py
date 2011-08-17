"""
RL Framework
Authors: Arun Chaganty
General utility functions
"""

import numpy as np

def normalise( dist ):
    return dist / np.sum( dist )
    
def choose( dist ):
    vs, dist = zip( *dist )
    dist = normalise( dist )
    idx = np.random.multinomial( 1, dist ).argmax()
    return vs[ idx ]

