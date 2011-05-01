"""
TaxiOptions Environment
"""

import numpy as np
import scipy
import scipy.sparse as sparse
import OptionEnvironment
import Taxi
import networkx as nx
import pdb

class TaxiOptions(Taxi.Taxi, OptionEnvironment.OptionEnvironment):
    # Environment Interface
    def __init__(self, spec, option_scheme='none', max_steps=500 ):
        """
        @spec - Specification (size, endpoints, barriers); either exactly
                specified in a file, or with numeric values in a list
        @option_scheme - optimal|random|betweenness
        @max_steps - Number of steps that need to taken
        """

        Taxi.Taxi.__init__( self, spec, max_steps )
        # Add options for all the optimal states
        if option_scheme == "none":
            self.set_options( [] )
        elif option_scheme == "manual":
            self.set_options( self.__get_manual_options() )
        elif option_scheme == "optimal":
            self.set_options( self.__get_optimal_options() )
        elif option_scheme == "small-world":
            self.set_options( self.__get_small_world_options() )
        elif option_scheme == "random":
            self.set_options( self.__get_random_options() )
        elif option_scheme == "betweenness":
            self.set_options(self.__get_betweenness_options() )
        else:
            raise NotImplemented() 

    def start(self):
        return OptionEnvironment.OptionEnvironment.start( self )

    def react(self, action):
        return OptionEnvironment.OptionEnvironment.react( self, action )
    
    def __make_option_from_paths( self, dest, paths ):
        if paths.has_key( dest ):
            paths.pop( dest )
        start = set( paths.keys() )
        stop = set( [dest] )
        policy = dict( [ (k,v[-2]) for k,v in paths.items() ] )

        option = OptionEnvironment.DeterministicOption( start, stop, policy )

        return option

    def __get_optimal_options( self ):
        # Reverse the graph to get the shortest paths to state
        gr = self.graph.reverse()
        options = []
        
        in_taxi = len( self.starts )
        road_size = len( self.road_map )
        for dest_i in xrange( len( self.starts ) ):
            # Paths to the passenger
            for start_i in xrange( len( self.starts ) ):
                if start_i == dest_i: 
                    continue
                st = self.get_state( start_i, dest_i, *self.starts[ start_i ] )
                paths = nx.shortest_path( gr, source=st )
                options.append( self.__make_option_from_paths( st, paths ) )
            # Paths to the destination
            grs = gr.subgraph( range( 
                self.get_state( in_taxi, dest_i, 0, 0 ),
                self.get_state( in_taxi, dest_i, road_size-1, road_size-1 ) + 1 ) )
            st = self.get_state( in_taxi, dest_i, *self.starts[ dest_i ] )
            paths = nx.shortest_path( grs, source=st )
            options.append( self.__make_option_from_paths( st, paths ) )

        return options

    def __get_manual_options( self ):
        """Use manually-defined options"""
        # Reverse the graph to get the shortest paths to state
        gr = self.graph.reverse()
        options = []
        
        in_taxi = len( self.starts )
        road_size = len( self.road_map )
        for dest_i in xrange( len( self.starts ) ):
            # Paths to the passenger
            for start_i in xrange( len( self.starts ) ):
                if start_i == dest_i: 
                    continue
                for start_j in xrange( len( self.starts ) ):
                    if start_j == dest_i: 
                        continue
                    # Paths to some pad
                    st = self.get_state( start_i, dest_i, *self.starts[ start_j ] )
                    paths = nx.shortest_path( gr, source=st )
                    options.append( self.__make_option_from_paths( st, paths ) )
            # Paths to the destination
            grs = gr.subgraph( range( 
                self.get_state( in_taxi, dest_i, 0, 0 ),
                self.get_state( in_taxi, dest_i, road_size, road_size-1 ) ) )
            for dest_j in xrange( len( self.starts ) ):
                st = self.get_state( in_taxi, dest_i, *self.starts[ dest_j ] )
                paths = nx.shortest_path( grs, source=st )
                options.append( self.__make_option_from_paths( st, paths ) )

        return options

    def __make_option_from_path( self, start, stop, path ):
        start = set( [start] )
        stop = set( [stop] )
        policy = dict( zip( path[:-1], path[1:] ) )

        option = OptionEnvironment.DeterministicOption( start, stop, policy )

        return option

    def __get_small_world_options( self, r = 2 ):
        # Get all the edges in the graph
        path_lengths = nx.shortest_path_length( self.graph )
        paths = nx.shortest_path( self.graph )

        options = []

        for node, node_dists in path_lengths.items():
            node_dists.pop( node )
            if not node_dists: 
                continue
            neighbours, dists = zip( *node_dists.items() )
            # Create a pr distribution
            dists = np.power( np.array( dists, dtype=float ), -r )
            # Zero out neighbours
            for i in xrange( len( dists ) ):
                if dists[i] == 1: dists[i] = 0
            if not dists.any(): 
                continue
            dists = dists / sum(dists)
            idx = np.random.multinomial(1, dists).argmax()

            dest = neighbours[ idx ]
            options.append( self.__make_option_from_path( node, dest, paths[node][dest] ) )

        return options

    def __get_random_options( self ):
        # Get all the edges in the graph
        paths = nx.shortest_path( self.graph )

        options = []

        for node, node_paths in paths.items():
            neighbours, node_paths = zip( *node_paths.items() )
            dist = np.ones( len( node_paths ) )
            # Zero out neighbours
            for i in xrange( len( dist ) ):
                if len( node_paths[i] ) < 2:
                    dist[i] = 0
            if not dist.any(): 
                continue
            dist = dist / sum(dist)
            idx = np.random.multinomial(1, dist).argmax()

            options.append( self.__make_option_from_path( node, neighbours[idx], node_paths[idx] ) )

        return options

    def __get_betweenness_options(self):
        #get a reverse graph( directed edgesd being reversed in direction)
        #calculate betweenness measure for the graph
        #search though graph and get nodes with have local maximas of betweenness 
        #arrange the nodes based on betweenness measure ( a list )
        #for each local maxima node:
        #     for all other nodes in the graph:
        #         find path to all these ( dictionary of dictionary )
        #augment the data structure by removing nodes removing paths with 
        #length smaller than minimum hop length and more than maximum hop length 
        #
        gr = self.graph.reverse()
        """
        for node in self.graph.nodes():
            print str(node) + " : " + str(self.graph[node])

        print " The reverse graph "
        for node in gr.nodes():
            print str(node) + " : " + str(gr[node])
        """
        options = []
        dict_betweenness_scores = nx.betweenness_centrality(self.graph, normalized=True, weighted_edges=False)
        #list of nodes which are local maximas
        local_maximas = self._get_local_maximas(self.graph, dict_betweenness_scores)
        local_maximas = self._sort_descending_betweenness_scores(local_maximas, dict_betweenness_scores)
        #the returned value is a dictionary of dictionary

        local_maxima_paths = self._get_paths_to_local_maximas(gr, local_maximas)
        #local_maxima_paths needs to be converted to a single dictionary 
        for local_maxima, paths in local_maxima_paths:
            options.append(self.__make_option_from_paths(local_maxima, paths))

        return options


    def _get_local_maximas(self, graph, dict_betweenness_scores):
        #return list of local_maximas
        local_maximas = []
        for node in graph.nodes():
            is_local_maxima = True
            for neighbor in graph.neighbors(node):
                if dict_betweenness_scores[neighbor] > dict_betweenness_scores[node]:
                    is_local_maxima = False
                    break 
            if is_local_maxima:
                local_maximas.append(node)

        return local_maximas

    def _sort_descending_betweenness_scores(self, local_maximas, dict_betweenness_scores):
        #sort the list of local_betweenness_maximum nodes
        #return sorted(reverse) list

        #not sure is this is an in 
        #place sort or it returns a sorted list
        #the sort function sorts the list in place
        local_maximas.sort(key=lambda node: dict_betweenness_scores[node])
        local_maximas = local_maximas[len(local_maximas) : 0 : -1]
        return local_maximas
        #local_maximas.reverse() # why does it return an empty list

        """
        mylist.sort(key=lambda x: x.lower())
        def mykey(adict): return adict['name']
        x = [{'name': 'Homer', 'age': 39}, {'name': 'Bart', 'age':10}]
        sorted(x, key=mykey)
        """
        
        
    #gr is a reverse graph: hence you go looking 
    #for paths from local maximas from local maximas to other nodes: am I correct ??
    def _get_paths_to_local_maximas(self, gr, local_maximas):
        #return a dictionary of dictionary of paths
        for local_maxima in local_maximas:
            paths = nx.shortest_path(gr, source=local_maxima)
            paths = dict( [ (node,path) for node,path in paths.items() if len(path) < 8 ] )
            yield local_maxima, paths
        return 

