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
    def state_idx( road_map, f, y, x ):
        """Compute the index of the state"""

        size = road_map.shape
        st, offset = x, size[1]
        st, offset = st + offset * y, offset * size[0]

        return f.get( st, None )

    @staticmethod
    def idx_state( road_map, g, state ):
        """Compute the state for the index"""
        state = g( state )
        x, state = state % size[1], state / size[1]
        y, state = state % size[0], state / size[0]

        return y, x

    @staticmethod
    def make_map_from_size( height, width ):
        raise NotImplemented()
        pass

    @staticmethod
    def make_map_from_txt_file( fname ):
        spec = map( str.strip, open( fname ).readlines() )
        size = tuple( map( int, spec[0].split() ) )

        def row_to_int( row ):
            return map( int, row.split() )
        road_map = np.array( map( row_to_int, spec[ 1: ] ) )

        if size != road_map.shape:
            raise ValueError()

        return road_map

    @staticmethod
    def make_map_from_tsv_file( fname ):
        spec = open( fname ).readlines()
        width = len(spec[0].split('\t'))
        height = len(spec)
        size = (height, width)

        def row_to_int( row ):
            row = row.split('\t')
            for i in xrange(len(row)):
                if row[i] == 'F': 
                    row[i] = 0
                else:
                    row[i] = 1
            return row
        road_map = np.array( map( row_to_int, spec ) )

        if size != road_map.shape:
            raise ValueError()

        return road_map

    @staticmethod
    def get_random_goal( road_map ):
        size = road_map.shape

        loc = np.random.randint( 0, size[0] ), np.random.randint( 0, size[1] ) 
        while road_map[ loc ] == Rooms.WALL:
            loc = np.random.randint( 0, size[0] ), np.random.randint( 0, size[1] ) 

        return loc

    @staticmethod
    def create_bijection( road_map ):
        size = road_map.shape
        S = size[ 0 ] * size[ 1 ]

        f = {}
        g = {}

        s_ = 0
        for y in xrange( size[ 0 ] ):
            for x in xrange( size[ 1 ] ):
                s, offset = x, size[1]
                s, offset = s + offset * y, offset * size[0]

                if road_map[y, x] == 0:
                    f[s] = s_
                    g[s_] = s
                    s_+=1
        return f, g 

    @staticmethod
    def make_mdp( road_map ):
        size = road_map.shape
        min_size = len( road_map[ road_map == 0] )
        f, g = Rooms.create_bijection( road_map )

        state_idx = functools.partial( Rooms.state_idx, road_map, f )

        goal = Rooms.get_random_goal( road_map )

        S = min_size
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
                if s is None: continue

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
        # Add rewards to all states that transit into the goal state
        s = state_idx( *goal )
        for s_ in xrange( S ):
            R[ (s_,s) ] = Rooms.REWARD_SUCCESS - Rooms.REWARD_BIAS
        
        start_set = None
        end_set = [ s ]

        return S, A, P, R, R_bias, start_set, end_set

    @staticmethod
    def create( spec ):
        """Create a room from @spec"""
        if spec is None:
            raise NotImplemented
        else:
            extn = spec.split('.')[-1]
            if extn == "tsv":
                road_map = Rooms.make_map_from_tsv_file( spec )
            else:
                road_map = Rooms.make_map_from_txt_file( spec )

        return Environment( Rooms, *Rooms.make_mdp( road_map ) )

    @staticmethod
    def reset_rewards( env, spec ):
        if spec is None:
            raise NotImplemented
        else:
            extn = spec.split('.')[-1]
            if extn == "tsv":
                road_map = Rooms.make_map_from_tsv_file( spec )
            else:
                road_map = Rooms.make_map_from_txt_file( spec )

        goal = Rooms.get_random_goal( road_map )
        f, g = Rooms.create_bijection( road_map )
        state_idx = functools.partial( Rooms.state_idx, road_map, f )

        # Reset the rewards
        R = {}
        # Add rewards to all states that transit into the goal state
        s = state_idx( *goal )
        for s_ in xrange( env.S ):
            R[ (s_,s) ] = Rooms.REWARD_SUCCESS - Rooms.REWARD_BIAS
        
        start_set = None
        end_set = [ s ]

        return Environment( Rooms, env.S, env.A, env.P, R, env.R_bias, start_set, end_set )
