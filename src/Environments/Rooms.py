"""
RL Framework
Authors: Arun Chaganty, Prateek Gaur
Rooms Environment
"""

import numpy as np
from Environment import *
import functools 

class Rooms():
    """
    Rooms Environment
    Expects specification file to be given
    """

    WALL        = 1
    GOAL        = 2

    MOVE_UP     = 0
    MOVE_DOWN   = 1
    MOVE_LEFT   = 2
    MOVE_RIGHT  = 3

    ACCURACY = 0.67

    REWARD_BIAS = -1
    REWARD_FAILURE = -20 - REWARD_BIAS
    REWARD_SUCCESS = 50 - REWARD_BIAS
    REWARD_CHECKPOINT = 0 # - REWARD_BIAS

    @staticmethod
    def state_idx( road_map, y, x ):
        """Compute the index of the state"""

        size = road_map.shape
        st, offset = x, size[1]
        st, offset = st + offset * y, offset * size[0]

        return st

    @staticmethod
    def idx_state( road_map, st ):
        """Compute the state for the index"""
        x, state = state % size[1], state / size[1]
        y, state = state % size[0], state / size[0]

        return y, x

    @staticmethod
    def make_map_from_size( height, width ):
        raise NotImplemented()
        pass

    @staticmethod
    def make_map_from_file( fname ):
        spec = map( str.strip, open( fname ).readlines() )
        size = tuple( map( int, spec[0].split() ) )

        def row_to_int( row ):
            return map( int, row.split() )
        road_map = np.array( map( row_to_int, spec[ 1: ] ) )

        if size != road_map.shape:
            raise ValueError()

        goal = ()
        for y in xrange( size[0] ):
            for x in xrange( size[1] ):
                if road_map[ y, x ] & Rooms.GOAL :
                    goal = (y,x)
                    break

        return road_map, goal

    @staticmethod
    def make_mdp( road_map, goal ):
        size = road_map.shape
        state_idx = functools.partial( Rooms.state_idx, road_map )

        S = size[ 0 ] * size[ 1 ]
        A = 4 # up down left right
        P = [ [ [] for i in xrange( S ) ] for j in xrange( A ) ]
        R = {}
        R_bias = Rooms.REWARD_BIAS

        # Populate the P table
        ACCURACY = Rooms.ACCURACY
        RESIDUE = (1.0 - ACCURACY)/3
        for y in xrange( size[ 0 ] ):
            for x in xrange( size[ 1 ] ):
                s = state_idx( y, x )

                if y > 0 and road_map[ y-1, x ] & Rooms.WALL == 0:
                    up_state = y-1, x
                else:
                    up_state = y, x
                if y + 1 < size[ 0 ] and road_map[ y+1, x ] & Rooms.WALL == 0:
                    down_state = y+1, x
                else:
                    down_state = y, x
                if x > 0 and road_map[ y, x-1 ] & Rooms.WALL == 0:
                    left_state = y, x-1
                else:
                    left_state = y, x
                if x + 1 < size[ 1 ] and road_map[ y, x+1 ] & Rooms.WALL == 0:
                    right_state = y, x+1
                else:
                    right_state = y, x

                P[ Rooms.MOVE_UP ][ s ] = [
                        ( state_idx( *up_state ), ACCURACY ),
                        ( state_idx( *down_state ), RESIDUE ),
                        ( state_idx( *left_state ), RESIDUE ),
                        ( state_idx( *right_state ), RESIDUE ), ]
                P[ Rooms.MOVE_DOWN ][ s ] = [
                        ( state_idx( *up_state ), RESIDUE ),
                        ( state_idx( *down_state ), ACCURACY ),
                        ( state_idx( *left_state ), RESIDUE ),
                        ( state_idx( *right_state ), RESIDUE ), ]
                P[ Rooms.MOVE_LEFT ][ s ] = [
                        ( state_idx( *up_state ), RESIDUE ),
                        ( state_idx( *down_state ), RESIDUE ),
                        ( state_idx( *left_state ), ACCURACY ),
                        ( state_idx( *right_state ), RESIDUE ), ]
                P[ Rooms.MOVE_RIGHT ][ s ] = [
                        ( state_idx( *up_state ), RESIDUE ),
                        ( state_idx( *down_state ), RESIDUE ),
                        ( state_idx( *left_state ), RESIDUE ),
                        ( state_idx( *right_state ), ACCURACY ), ]
        # Remove actions from the goal state
        s = state_idx( *goal )
        for a in xrange( A ):
            P[ a ][ s ] = []

        # Add rewards to all states that transit into the goal state
        for s_ in xrange( S ):
            R[ (s_,s) ] = Rooms.REWARD_SUCCESS - Rooms.REWARD_BIAS
        
        start_set = None

        return S, A, P, R, R_bias, start_set

    @staticmethod
    def create( spec ):
        """Create a room from @spec"""
        if spec is None:
            road_map, starts = Rooms.make_map_from_size( 5, 5 )
        else:
            road_map, goal = Rooms.make_map_from_file( spec )
        return Environment( *Rooms.make_mdp( road_map, goal ) )

