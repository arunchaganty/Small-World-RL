"""
RL Framework
Authors: Arun Chaganty, Prateek Gaur
Taxi Environment
"""

import numpy as np
from Environment import *
import functools 

class Taxi():
    """
    Taxi Environment
    Expects specification (size, endpoints, barriers) to be given
    """

    NONE = 0
    TOP_DOWN = 1
    LEFT_RIGHT = 2 
    STOP = 4

    MOVE_UP     = 0
    MOVE_DOWN   = 1
    MOVE_LEFT   = 2
    MOVE_RIGHT  = 3
    MOVE_PICK   = 4
    MOVE_DROP   = 5

    ACCURACY = 0.8

    REWARD_BIAS = -1
    REWARD_FAILURE = -20 - REWARD_BIAS
    REWARD_SUCCESS = 50 - REWARD_BIAS
    REWARD_CHECKPOINT = 0 # - REWARD_BIAS

    @staticmethod
    def state_idx( road_map, starts, completed, in_taxi, pasn, dest, y, x ):
        """Compute the index of the state"""

        size = road_map.shape
        STARTS = len( starts )
        st, offset = x, size[1]
        st, offset = st + offset * y, offset * size[0]
        st, offset = st + offset * dest, offset * STARTS

        if completed:
            st = offset = offset * STARTS
        elif in_taxi:
            st, offset = st + offset * (STARTS-1), offset * STARTS
        elif pasn < dest:
            st, offset = st + offset * pasn, offset * STARTS
        elif pasn > dest:
            st, offset = st + offset * (pasn - 1), offset * STARTS
        else:
            raise ValueError()

        return st

    @staticmethod
    def idx_state( road_map, st ):
        """Compute the state for the index"""
        x, state = state % size[1], state / size[1]
        y, state = state % size[0], state / size[0]
        dest, state = state % STARTS, state / STARTS
        pasn, state = state % STARTS, state / STARTS
        completed = bool( state )

        return completed, in_taxi, pasn, dest, y, x

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

        starts = []
        for y in xrange( size[0] ):
            for x in xrange( size[1] ):
                if road_map[ y, x ] & Taxi.STOP :
                    starts.append( (y,x) )

        return road_map, starts

    @staticmethod
    def make_mdp( road_map, starts ):
        size = road_map.shape
        STARTS = len( starts )
        state_idx = functools.partial( Taxi.state_idx, road_map, starts )

        def make_map( road_map, in_taxi, pasn, dest, P ): 
            state_idx_ = functools.partial( state_idx, False, in_taxi, pasn, dest )

            def make_move( axis, y, x ):
                ACCURACY = Taxi.ACCURACY
                RESIDUE = 1.0 - ACCURACY
                moves = []

                if axis == Taxi.MOVE_UP:
                    moves.append( (state_idx_( y-1, x ), ACCURACY) )
                elif axis == Taxi.MOVE_DOWN:
                    moves.append( (state_idx_( y+1, x ), ACCURACY) )
                elif axis == Taxi.MOVE_LEFT:
                    moves.append( (state_idx_( y, x-1 ), ACCURACY) )
                elif axis == Taxi.MOVE_RIGHT:
                    moves.append( (state_idx_( y, x+1 ), ACCURACY) )

                if axis == Taxi.MOVE_UP or axis == Taxi.MOVE_DOWN:
                    possibles = []
                    if x > 0 and road_map[ y, x-1 ] & Taxi.LEFT_RIGHT == 0:
                        possibles.append( state_idx_( y, x-1 ) )
                    if x + 1 < size[1] and road_map[ y, x ] & Taxi.LEFT_RIGHT == 0:
                        possibles.append( state_idx_( y, x+1 ) )
                    if len( possibles ):
                        moves += [ (s, RESIDUE / len( possibles ) ) for s in possibles ]
                elif axis == Taxi.MOVE_LEFT or axis == Taxi.MOVE_RIGHT:
                    possibles = []
                    if y > 0 and road_map[ y-1, x ] & Taxi.TOP_DOWN == 0:
                        possibles.append( state_idx_( y-1, x ) )
                    if y + 1 < size[0] and road_map[ y, x ] & Taxi.TOP_DOWN == 0:
                        possibles.append( state_idx_( y+1, x ) )
                    if len( possibles ):
                        moves += [ (s, RESIDUE / len( possibles ) ) for s in possibles ]

                return moves

            for y in xrange( size[ 0 ] ):
                for x in xrange( size[ 1 ] ):
                    s = state_idx_( y, x )
                    if y > 0 and road_map[ y-1, x ] & Taxi.TOP_DOWN == 0:
                        P[ Taxi.MOVE_UP ][ s ] += make_move( 0, y, x )
                    if y + 1 < size[0] and road_map[ y, x ] & Taxi.TOP_DOWN == 0:
                        P[ Taxi.MOVE_DOWN ][ s ] += make_move( 1, y, x )
                    if x > 0 and road_map[ y, x-1 ] & Taxi.LEFT_RIGHT == 0:
                        P[ Taxi.MOVE_LEFT ][ s ] += make_move( 2, y, x )
                    if x + 1 < size[1] and road_map[ y, x ] & Taxi.LEFT_RIGHT == 0:
                        P[ Taxi.MOVE_RIGHT ][ s ] += make_move( 3, y, x )
            return P

        # Create P, R
        S = size[ 0 ] * size[ 1 ] * STARTS * STARTS + 1
        A = 6 # up down left right pick drop
        P = [ [ [] for i in xrange( S ) ] for j in xrange( A ) ]
        R = {}
        R_bias = Taxi.REWARD_BIAS

        in_taxi = False
        for pasn in xrange( STARTS ):
            for dest in xrange( STARTS ):
                if pasn == dest:
                    continue
                P = make_map( road_map, in_taxi, pasn, dest, P )
                # Add pickup actions
                y, x = starts[ pasn ]
                s = state_idx( False, False, pasn, dest, y, x )
                s_ = state_idx( False, True, pasn, dest, y, x )
                P[ Taxi.MOVE_PICK ][ s ] += [ (s_, 1.0), ]
                R[ (s, s_) ] = Taxi.REWARD_CHECKPOINT

        in_taxi = True
        pasn = -1 
        for dest in xrange( STARTS ):
            P = make_map( road_map, in_taxi, pasn, dest, P )
            y, x = starts[ dest ]
            s = state_idx( False, True, pasn, dest, y, x )
            s_ = state_idx( True, True, pasn, dest, y, x )
            P[ Taxi.MOVE_DROP ][ s ] += [ (s_, 1.0), ]
            R[ (s, s_) ] = Taxi.REWARD_SUCCESS

        start_set = range( 
                state_idx( False, False, 1, 0, y, x ),
                state_idx( False, False, 2, 3, y, x ) + 1 )
        end_set = [ state_idx( True, True, 0, 0, 0, 0 ) ]

        return S, A, P, R, R_bias, start_set, end_set

    @staticmethod
    def create( spec ):
        """Create a taxi from @spec"""
        if spec is None:
            road_map, starts = Taxi.make_map_from_size( 5, 5 )
        else:
            road_map, starts = Taxi.make_map_from_file( spec )
        return Environment( Taxi, *Taxi.make_mdp( road_map, starts ) )

    @staticmethod
    def reset_rewards( env, *args ):
        return env

