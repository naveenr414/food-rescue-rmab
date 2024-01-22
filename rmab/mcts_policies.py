import numpy as np

from rmab.uc_whittle import Memoizer
from rmab.omniscient_policies import whittle_greedy_policy, whittle_index

from copy import deepcopy
from mcts import mcts
import time 

class VolunteerState():
    def __init__(self):
        self.non_match_prob = 1
        self.previous_pulls = set()
        self.N = 0
        self.arm_values = []
        self.budget = 0
        self.states = []
        self.discount = 0.9
        self.lamb = 0
        self.max_depth = 1
        self.rollout_number = 1
        self.previous_rewards = []
        self.transitions = []
        self.cohort_size = 0
        self.volunteers_per_arm = 0
        self.memory = {}
        self.memoizer = Memoizer('optimal')

    def getPossibleActions(self):
        possibleActions = []
        for i in range(self.N):
            if i not in self.previous_pulls:
                possibleActions.append(Action(arm=i))
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.previous_pulls.add(action.arm)
        newState.memoizer = self.memoizer

        return newState

    def isTerminal(self):
        if len(self.previous_pulls) == self.budget:
            return True 
        return False         

    def getReward(self):
        """Compute the reward using the Whittle index + Matching
            For each combo"""

        policy = 'index'

        if policy == 'reward':
            non_match_prob = 1 
            for i in self.previous_pulls:
                expected_active_rate = 0
                for i in range(len(self.states)):
                    action = int(i in self.previous_pulls)
                    expected_active_rate += self.transitions[i//self.volunteers_per_arm,self.states[i],action,1]

                non_match_prob *= (1-self.states[i]*self.arm_values[i])

            value = 1-non_match_prob  + self.lamb*expected_active_rate

        elif policy == 'index':
            state_WI = whittle_index(self.env,self.states,self.budget,self.lamb,self.memoizer,reward_function="activity")
            non_match_prob = 1 

            for i in self.previous_pulls:
                non_match_prob *= (1-self.env.transitions[i//self.volunteers_per_arm,self.states[i],1,1]*self.arm_values[i])

            value = (1-non_match_prob)/(1-self.discount) + self.lamb*np.sum(state_WI[list(self.previous_pulls)]) 

        return value 

    def set_state(self,states):
        self.states = states

class VolunteerStateTwoStepMCTS(VolunteerState):
    def __init__(self):
        super().__init__()

        self.num_by_cohort = []
        self.current_cohort = 0
        self.num_in_cohort = 0

    def set_num_by_cohort(self,num_by_cohort):
        """Each level in the MCTS search is a different group  
            that we aim to look at
            The current cohort is the current group we're looking at"""

        self.num_by_cohort = num_by_cohort
        self.current_cohort = 0
        while self.num_by_cohort[self.current_cohort] == 0:
            self.current_cohort += 1

    def getPossibleActions(self):
        possibleActions = []
        for i in range(self.volunteers_per_arm):
            idx = self.current_cohort*self.volunteers_per_arm + i
            if idx not in self.previous_pulls:
                possibleActions.append(Action(arm=idx))
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.previous_pulls.add(action.arm)
        newState.memoizer = self.memoizer
        newState.num_in_cohort += 1
        if newState.num_in_cohort >= newState.num_by_cohort[newState.current_cohort]:
            newState.num_in_cohort = 0
            newState.current_cohort += 1
            while newState.current_cohort < len(
                newState.num_by_cohort) and newState.num_by_cohort[newState.current_cohort] == 0:
                newState.current_cohort += 1

        return newState


class Action():
    def __init__(self, arm):
        self.arm = arm 

    def __str__(self):
        return str(self.arm)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.arm == other.arm

    def __hash__(self):
        return hash((self.arm))

def set_initial_state(initialState,env,state,memory,lamb):
    """Set the hyperparameter values for the initial state
        based on environment + state information
        
    Arguments:
        initialState: VolunterState or VolunterStateTwoStep
        env: Simulator Environmenet
        state: Numpy array of the current state
        memory: List with the first element being a dictionary
            The second being a memoizer
        lamb: Float, the current \lambda value
            
    Returns: The initialState, with all the parameters set"""

    initialState.arm_values = np.array(env.match_probability_list)[env.agent_idx] 
    initialState.N = len(initialState.arm_values)
    initialState.budget = env.budget 
    initialState.transitions = env.transitions
    initialState.cohort_size = env.cohort_size
    initialState.volunteers_per_arm = env.volunteers_per_arm
    initialState.discount = env.discount 
    initialState.env = env 
    initialState.lamb = lamb
    initialState.set_state(state)
    initialState.memory = memory[0]
    initialState.memoizer = memory[1]

    return initialState

def one_step_idx_to_action(selected_idx,env,state,lamb,memory):
    """Turn a list of volunteers to notify into a 0-1 
        vector for action
        
    Arguments:
        selected_idx: Numpy array of volunteers to notify
        env: RMAB Simulator Environment
        state: 0-1 numpy array for the state
        lamb: Float, what \lambda value
        memory: List with the first element as a dictionary
            second element as a memoizer
            
    Returns: 0-1 numpy array with the actions to take"""

    selected_idx = np.array(selected_idx)
    action = np.zeros(len(state), dtype=np.int8)
    action[selected_idx] = 1

    return action, memory

def two_step_mcts_to_action(env,state,lamb, memory): 
    """Run MCTS to find the groups, then run MCTS again to find
        which volunteers in each group to notify
        
    Arguments:
        selected_idx: Numpy array of volunteers to notify
        env: RMAB Simulator Environment
        state: 0-1 numpy array for the state
        lamb: Float, what \lambda value
        memory: List with the first element as a dictionary
            second element as a memoizer
            
    Returns: 0-1 numpy array with the actions to take"""


    fractions = [1/3,1/2]
    fractions.append((1-sum(fractions))/2)
    greedy_action, _ = mcts_policy(env,state,env.budget,lamb,memory,None,timeLimit=env.TIME_PER_RUN * fractions[0])
    bin_counts = np.zeros(env.cohort_size, dtype=int)
    for i in range(env.cohort_size):
        bin_counts[i] = np.sum(greedy_action[i*env.volunteers_per_arm:(i+1)*env.volunteers_per_arm])
    num_by_cohort = bin_counts

    initialState = VolunteerStateTwoStepMCTS()
    initialState = set_initial_state(initialState,env,state,memory,lamb)
    initialState.set_num_by_cohort(num_by_cohort)
    
    searcher = mcts(timeLimit=env.TIME_PER_RUN * fractions[1]) 
    selected_idx = []
    current_state = initialState

    for _ in range(env.budget):
        action = searcher.search(initialState=current_state)
        current_state = current_state.takeAction(action)
        selected_idx.append(action.arm)
        searcher = mcts(timeLimit=env.TIME_PER_RUN * fractions[2])

    action = np.zeros(len(state), dtype=np.int8)
    action[selected_idx] = 1

    return action, memory

def two_step_mcts_to_whittle(env,state,lamb, memory): 
    """Run Whittle+Greedy to find the groups, then run MCTS again to find
        which volunteers in each group to notify
        
    Arguments:
        selected_idx: Numpy array of volunteers to notify
        env: RMAB Simulator Environment
        state: 0-1 numpy array for the state
        lamb: Float, what \lambda value
        memory: List with the first element as a dictionary
            second element as a memoizer
            
    Returns: 0-1 numpy array with the actions to take"""


    fractions = [1/2]
    fractions.append((1-sum(fractions))/2)
    greedy_action, _ = whittle_greedy_policy(env,state,env.budget,lamb,memory[1],None)
    bin_counts = np.zeros(env.cohort_size, dtype=int)
    for i in range(env.cohort_size):
        bin_counts[i] = np.sum(greedy_action[i*env.volunteers_per_arm:(i+1)*env.volunteers_per_arm])
    num_by_cohort = bin_counts

    initialState = VolunteerStateTwoStepMCTS()
    initialState = set_initial_state(initialState,env,state,memory,lamb)

    initialState.set_num_by_cohort(num_by_cohort)
    
    searcher = mcts(timeLimit=env.TIME_PER_RUN * fractions[0]) 
    selected_idx = []
    current_state = initialState 

    for _ in range(env.budget):
        action = searcher.search(initialState=current_state)
        current_state = current_state.takeAction(action)
        selected_idx.append(action.arm)
        searcher = mcts(timeLimit=env.TIME_PER_RUN * fractions[1])

    action = np.zeros(len(state), dtype=np.int8)
    action[selected_idx] = 1

    return action, memory

def init_memory(memory):
    """For MCTS process, we store two things in the memory
        a) A dictionary with past iterations which we've solved
        b) A memoizer for the Whittle process
    
    Arguments:
        memory: None or a list with a dictionary and a memoizer
    
    Returns: A list with a dictionary and a memoizer"""

    if memory == None:
        memory = [{}, Memoizer('optimal')]
    return memory 

def run_two_step(env,state,budget,lamb,memory,per_epoch_results,idx_to_action,timeLimit=-1):
    """Run a two-step policy using the idx_to_action function
    
    Arguments: 
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Information on previously computed Whittle indices, the memoizer
        per_epoch_results: Any information computed per epoch; unused here
        idx_to_action: Function such as two_step_mcts_to_whittle
        timeLimit: Maximum time, in miliseconds, for all MCTS calls
            By default this should be env.TIME_PER_RUN
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=
        List with a dictionary and a memoizer"""
    
    memory = init_memory(memory)
    if timeLimit == -1:
        timeLimit = env.TIME_PER_RUN 

    if ''.join([str(i) for i in state]) in memory[0]:
        return memory[0][''.join([str(i) for i in state])], memory

    action, memory = idx_to_action(env,state,lamb,memory)
    memory[0][''.join([str(i) for i in state])] = action 
    return action, memory

def run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts_function,timeLimit=-1):
    """Compute an MCTS-based policy which selects arms to notify 
    sequentially, then rolls out for Q=5 steps to predict reward

    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Information on previously computed Whittle indices, the memoizer
        per_epoch_results: Any information computed per epoch; unused here
        timeLimit: Total time for all MCTS calls 
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    memory = init_memory(memory)

    if ''.join([str(i) for i in state]) in memory[0]:
        return memory[0][''.join([str(i) for i in state])], memory
    
    if timeLimit == -1:
        timeLimit = env.TIME_PER_RUN 

    initialState = VolunteerState()
    initialState = set_initial_state(initialState,env,state,memory,lamb)

    fraction_first_budget = 4/5
    fraction_other = (1-fraction_first_budget)/(budget-1)

    searcher = mcts_function(timeLimit=timeLimit*fraction_first_budget)
    
    selected_idx = []
    current_state = initialState 

    for _ in range(budget):
        action = searcher.search(initialState=current_state)
        current_state = current_state.takeAction(action)
        selected_idx.append(action.arm)
        searcher = mcts_function(timeLimit=timeLimit*fraction_other)

    action, memory = one_step_idx_to_action(selected_idx,env,state,lamb,memory)
    memory[0][''.join([str(i) for i in state])] = action 
    return action, memory

def mcts_policy(env,state,budget,lamb,memory,per_epoch_results,num_iterations=100,timeLimit=-1):
    return run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts,timeLimit=timeLimit)

def mcts_mcts_policy(env,state,budget,lamb,memory,per_epoch_results):
    return run_two_step(env,state,budget,lamb,memory,per_epoch_results,two_step_mcts_to_action)

def mcts_whittle_policy(env,state,budget,lamb,memory,per_epoch_results):
    return run_two_step(env,state,budget,lamb,memory,per_epoch_results,two_step_mcts_to_whittle)
