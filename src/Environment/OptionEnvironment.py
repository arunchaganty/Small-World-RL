"""
Environment that supports Options
"""

import Environment
import pdb

class Option:
    """Encapsulates an option: start, stop and policy predicates"""
    def __init__(self, start, stop, policy):
        """Create an option with the start predicate, stop predicate and policy
        predicate"""
        self.start = start
        self.stop = stop
        self.policy = policy
    
    def can_start(self, state):
        """Check if the option can be be started in this state"""
        return self.start( state )

    def should_stop(self, state):
        """Check if the option should be stopped"""
        return self.stop( state )

    def execute(self, state, actions ):
        """Choose an option as per the policy"""
        return self.policy( state, actions )

class MapOption( Option ):
    def can_start(self, state):
        """Check if the option can be be started in this state"""
        return bool( self.start[ state ] )

    def should_stop(self, state):
        """Check if the option should be stopped"""
        return bool( np.random.binomial(1, self.stop[ state ] ) )

    def execute(self, state, actions ):
        """Choose an option as per the policy"""
        return np.random.multinomial(1, self.policy[ state, : ] ).argmax()

class DeterministicOption( Option ):
    def can_start(self, state):
        """Check if the option can be be started in this state"""
        return state in self.start

    def should_stop(self, state):
        """Check if the option should be stopped"""
        return state in self.stop

    def execute(self, state, actions ):
        """Choose an option as per the policy"""
        action = self.policy[ state ]
        assert( action in actions )
        return action

class OptionEnvironment(Environment.Environment):
    """Environment that defines a graph structure"""
    options = []
    __last_state_action = None

    def __init__( self ):
        Environment.Environment.__init__(self)

    def set_options( self, options ):
        """Set the options to be used by the environment"""
        self.options = options

    def get_options( self, state ):
        """Get all relevant options for the state"""
        return [ option for option in self.options if 
                option.can_start( state ) ]

    def start(self):
        """Initialise the Environment
        @returns initial state and valid actions
        """
        state, actions = self._start()
        self.__last_state_action = state, actions
        return state, actions + self.get_options( state )

    def react(self, action):
        """React to action
        @returns new state and valid actions, and reward, and if episode has
        ended
        """
        if isinstance( action, Option ):
            option = action
            history = []
            rewards = []
            state, actions = self.__last_state_action

            # Get the action from the option
            action = option.execute( state, actions )
            state, actions, reward, episode_ended = self._react( action )

            # Quit if the episode has ended
            if episode_ended:
                return state, actions + self.get_options( state ), reward, episode_ended 
                
            history.append( (state, actions) )
            rewards.append( reward )

            while not option.should_stop( state ):
                # Get the action from the option
                action = option.execute( state, actions )
                state, actions, reward, episode_ended = self._react( action )
                history.append( (state, actions) )
                rewards.append( reward )

                # Quit if the episode has ended
                if episode_ended: 
                    break

            self.__last_state_action = state, action
            return history, actions + self.get_options( state ), rewards, episode_ended 
                
        else:
            state, actions, reward, episode_ended = self._react( action )
            self.__last_state_action = state, actions
            return state, actions + self.get_options( state ), reward, episode_ended

