"""
Generate options
"""

import copy
import pickle
import random
import itertools
import numpy as np
import networkx as nx

import util
import Runner
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

def choose_small_world( path_lengths, s, r ):
    # Check distances
    dists = path_lengths[s][1]
    if s in dists: dists.pop( s )
    if not dists: return None

    neighbours, dists = zip( *dists.items() )
    # Create a pr distribution
    dists = np.power( np.array( dists, dtype=float ), -r )
    # Zero out neighbours
    for i in xrange( len( dists ) ):
        if dists[i] == 1: dists[i] = 0

    if not dists.any(): 
        return None

    s_ = util.choose( zip( neighbours, dists ) )
    return s_

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

def optimal_options_from_random_nodes( env, count ):
    """Create an option that takes a state to a random set of nodes"""
    g = env.to_graph()
    gr = g.reverse()

    nodes = gr.nodes()
    random.shuffle( nodes )
    options = util.progressMap( lambda n: optimal_point_option( g, gr, n, 16 ), nodes[:count] )

    return options

def optimal_options_from_betweenness( env, count ):
    g = env.to_graph()
    gr = g.reverse()

    """Create an option that takes a state to a random set of nodes"""
    maximas = find_betweenness_maxima( g )
    options = util.progressMap( lambda n: optimal_point_option( g, gr, n, 16 ), maximas[ :count ] )

    return options

def optimal_options_from_random_paths( env, count ):
    """Create an option that takes a state to a random set of nodes"""
    g = env.to_graph()
    gr = g.reverse()

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

def optimal_options_from_small_world( env, count, r ):
    """Create an option that takes a state to a random nodes as per a power-law dist"""
    g = env.to_graph()
    gr = g.reverse()

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
        s_ = choose_small_world( path_lengths, s, r )
        if not s_: continue
        paths.append( (s,s_) )

    options = util.progressMap( lambda (node, dest): optimal_path_option( g, gr, node, dest ), paths )
    return options

def learn_point_option( env, s, epochs, agent_type, agent_args ):
    """Learn an option to a state"""
    # Reset the rewards of the environment to reward the 
    env.R = {}
    for s_ in xrange( env.S ):
        env.R[ (s_,s) ] = env.domain.REWARD_SUCCESS - env.domain.REWARD_BIAS
    
    agent = agent_type( env.Q, *agent_args )
    Runner.run( env, agent, epochs )
    pi = agent.greedy_policy()

    I = set( pi.keys() )
    I.remove( s )
    B = { s: 1.0 }

    return Option( I, pi, B )

def learn_path_option( env, start, dest, epochs, agent_type, agent_args ):
    """Learn an option to a state"""
    # Reset the rewards of the environment to reward the 
    env.R = {}
    env.start_set = set([start])
    env.end_set = set([dest])
    for s_ in xrange( env.S ):
        env.R[ (s_,dest) ] = env.domain.REWARD_SUCCESS - env.domain.REWARD_BIAS
    
    agent = agent_type( env.Q, *agent_args )
    Runner.run( env, agent, epochs )
    pi = agent.greedy_policy()

    I = set( [start] )
    I.remove( dest )
    B = { state: 1.0 }

    return Option( I, pi, B )

def learn_option_from_policy( pi, Q, s, s_ ):
    """Extract an option from s to s_ from pi"""
    # The sub-policy such that Q(t,pi(t)) < Q(s_,pi(s_))
    pi_ = dict( [ (t,a) for (t,a) in pi.items() if Q[t][a] < Q[s_][pi[a]] ] )
    I = set([s])
    B = { s_ : 1.0 }

    return Option( I, pi_, B )

def learn_options_from_betweenness( epoch_budget, count, env, env_args, agent_type, agent_args ):
    """Create an option that takes a state to a random set of nodes"""
    g = env.to_graph()

    maximas = find_betweenness_maxima( g )[:count]
    # Evenly divide the epoch budget
    epochs = epoch_budget / len(maximas)

    options = util.progressMap( lambda n: learn_point_option( env, n, epochs, agent_type, agent_args ), maximas )

    return options

def learn_options_from_small_world( epoch_budget, count, env, env_args, agent_type, agent_args, r, searches = 20, beta = 1.1 ):
    """
    Learn options according to the small world distribution
    @r - exponent
    @alpha - Proportion taken each time
    """

    g = env.to_graph()
    gr = g.reverse()
    S = range( len( g.nodes() ) )

    # Get all the edges in the graph
    max_length = np.power( 16, 1.0/r ) # fn of r
    path_lengths = nx.all_pairs_shortest_path_length( g, cutoff=max_length ).items()


    def extract_small_world_options( pi, Q, r ):
        """Extract n options from pi according to the small world distribution"""
        # Choose a state at random
        random.shuffle( S )
        
        for s in S:
            # Choose a s_ ~ P_r(s) if Q(s_,pi(s_)) > Q(s, pi(s))
            s_ = choose_small_world( path_lengths, s, r )
            if not s_: continue
            if Q[s_][pi[s_]] > Q[s][pi[s]]:
                yield learn_option_from_policy( pi, Q, s, s_ )

    progress = ProgressBar( 0, count, mode='fixed' )
    oldprog = str(progress)

    options = []

    alpha = 1/float(searches)
    count_ = int(alpha*beta*count)
    # Evenly divide the epoch budget
    epochs = epoch_budget / searches
    for i in xrange( searches ):
        # Run an agent
        env = env.domain.reset_rewards( env, *env_args )
        agent = agent_type( env.Q, *agent_args ) 
        Runner.run( env, agent, epochs )

        # Extract a policy
        pi = agent.greedy_policy()
        options += list( itertools.islice( extract_small_world_options( pi, agent.Q, r ), count_ ) )

        # print progress
        progress.update_amount( len(options) )
        if oldprog != str(progress):
            print progress, "\r",
            sys.stdout.flush()
            oldprog=str(progress)
    print "\n"

    # We may have learnt a few extra, but that's ok; pick a random
    # @count of them
    random.shuffle( options )

    return options[:count]

def options_from_file( fname ):
    """Load options from a file"""
    options = pickle.load( fname )
    return options

