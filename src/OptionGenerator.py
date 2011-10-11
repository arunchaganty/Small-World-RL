"""
Generate options
"""

import pickle
import random
import numpy as np
import networkx as nx

import util
from Environment import *

def find_betweenness_maxima( g ):
    # Get betweenness scores
    bw = nx.betweenness_centrality( g )

    local_maximas = []
    for n, b in bw.items():
        for n_ in g.successors( n ):
            if bw[ n_ ] > b: break
        else:
            local_maximas.append( (n, b) )
    local_maximas.sort( key = lambda (x,v): -v )
    return [ x for (x,v) in local_maximas ]

# Optimal Primitives 
def optimal_point_option( g, gr, dest, max_length ):
    """Create an option that takes all connected states to dest"""
    paths = nx.predecessor(gr, source=dest, cutoff = max_length)

    I = set( paths.keys() )
    I.remove( dest )
    pi = {}
    for src, succ in paths.items():
        if src == dest: continue
        # Next link in the path
        succ = succ[ 0 ]

        # Choose the maximum probability action for this edge
        actions = [ (attrs['action'], attrs['pr']) for src, succ_, attrs in g.edges( src, data=True ) if succ_ == succ ] 
        action = max( actions, key = lambda (a,pr): pr )[ 0 ]

        pi[ src ] = ((action, 1.0),)
    B = { dest : 1.0 }
    
    return Option( I, pi, B )

def optimal_path_option( g, gr, start, dest, length = None ):
    """Create an option that takes a state to a dest"""
    # HACK (using + 3)
    if length == None:
        length = nx.shortest_path_length(g, source=start, target=dest) 
    max_length = length + 2

    o = optimal_point_option( g, gr, dest, max_length )
    # Start not reachable from dest
    if not start in o.I:
        return None
    else:
        o.I = set( [start] )

    return o

def optimal_options_from_random_nodes( g, gr, count ):
    """Create an option that takes a state to a random set of nodes"""
    nodes = gr.nodes()
    random.shuffle( nodes )
    options = util.progressMap( lambda n: optimal_point_option( g, gr, n, 16 ), nodes[:count] )

    return options

def optimal_options_from_betweenness( g, gr, count ):
    """Create an option that takes a state to a random set of nodes"""
    maximas = find_betweenness_maxima( g )
    options = util.progressMap( lambda n: optimal_point_option( g, gr, n, 16 ), maximas[ :count ] )

    return options

def optimal_options_from_random_paths( g, gr, count ):
    """Create an option that takes a state to a random set of nodes"""
    # Get all the edges in the graph
    nodes = g.nodes()
    random.shuffle( nodes )

    paths = []
    for node in nodes:
        if len( paths ) > count: 
            break
        neighbours = g.successors( node )
        if len( neighbours ) == 0: 
            continue
        dest = random.choice( neighbours )
        paths.append( (node, dest) )

    options = util.progressMap( lambda (node, dest): optimal_path_option( g, gr, node, dest ), paths )
    return options

def optimal_options_from_small_world( g, gr, count, r ):
    """Create an option that takes a state to a random nodes as per a power-law dist"""
    S = len( g.nodes() )
    states = range(S)

    # Get all the edges in the graph
    max_length = np.power( 16, 1.0/r ) # fn of r
    path_lengths = nx.all_pairs_shortest_path_length( g, cutoff=max_length ).items()

    random.shuffle(states)

    paths = []
    for s in states:
        if len( paths ) > count: 
            break
        dists = path_lengths[s][1]
        dists.pop( s )
        if not dists: 
            continue

        neighbours, dists = zip( *dists.items() )
        # Create a pr distribution
        dists = np.power( np.array( dists, dtype=float ), -r )
        # Zero out neighbours
        for i in xrange( len( dists ) ):
            if dists[i] == 1: dists[i] = 0
        if not dists.any(): 
            continue
        s_ = util.choose( zip( neighbours, dists ) )
        paths.append( (s,s_) )

    options = util.progressMap( lambda (node, dest): optimal_path_option( g, gr, node, dest ), paths )
    return options

def learn_point_option( env, state, epochs, agent_type, agent_args ):
    """Learn an option to a state"""
    # Reset the rewards of the environment to reward the 
    env.R = {}
    for s_ in xrange( env.S ):
        env.R[ (s_,s) ] = env.REWARD_SUCCESS - env.REWARD_BIAS
    
    agent = agent_type( env.Q, *agent_args )
    Runner.run( env, agent, epochs )
    pi = agent.greedy_policy()

    I = set( pi.keys() )
    I.remove( dest )
    B = { state: 1.0 }

    return Option( I, pi, B )

def learn_path_option( env, start, dest, epochs, agent_type, agent_args ):
    """Learn an option to a state"""
    # Reset the rewards of the environment to reward the 
    env.R = {}
    env.start_set = set([start])
    env.end_set = set([dest])
    for s_ in xrange( env.S ):
        env.R[ (s_,dest) ] = env.REWARD_SUCCESS - env.REWARD_BIAS
    
    agent = agent_type( env.Q, *agent_args )
    Runner.run( env, agent, epochs )
    pi = agent.greedy_policy()

    I = set( [start] )
    I.remove( dest )
    B = { state: 1.0 }

    return Option( I, pi, B )

def extract_options( policy ):
    pass

def options_from_file( fname ):
    """Load options from a file"""
    options = pickle.load( fname )
    return options

