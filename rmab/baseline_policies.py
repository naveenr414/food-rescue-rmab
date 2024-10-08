import numpy as np
from rmab.utils import custom_reward, one_hot, binary_to_decimal
from rmab.compute_whittle import arm_value_iteration_exponential

import random 

def greedy_policy(env,state,budget,lamb,memory,per_epoch_results):
    """Greedy policy that selects the budget highest values
        of state*match_probability + activity_score * lamb
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""
    N = len(state)

    if memory is None:
        p_matrix = compute_p_matrix(env,N)
    else:
        p_matrix = memory 

    score_by_agent = [p_matrix[i][state[i]] for i in range(N)]
    selected_idx = np.argsort(score_by_agent)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, p_matrix 

def random_policy(env,state,budget,lamb,memory, per_epoch_results):
    """Random policy that randomly notifies budget arms
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""


    N = len(state)
    selected_idx = random.sample(list(range(N)), budget)
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, None

def q_iteration_policy(env,state,budget,lamb,memory,per_epoch_results):
    """Q Iteration policy that computes Q values for all combinations of states
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: The Q Values
        debug: If we want to see the actual Q values, boolean
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    debug = False  
    
    Q_vals = per_epoch_results
    
    N = len(state)

    state_rep = binary_to_decimal(state)

    if debug: 
        for i in range(2**len(state)):
            state = [int(j) for j in bin(i)[2:].zfill(N)]
            max_action = np.argmax(Q_vals[i])
            value = Q_vals[i][max_action]
            action = [int(j) for j in bin(max_action)[2:].zfill(N)]
            print("State {} value {}".format(state,value))

    max_action = np.argmax(Q_vals[state_rep])
    binary_val = bin(max_action)[2:].zfill(N)

    action = np.zeros(N, dtype=np.int8)
    action = np.array([int(i) for i in binary_val])

    return action, None



def q_iteration_epoch(env,lamb,reward_function='combined'):
    """Compute Q Values for all combinations of agents in a given environment
    
    Arguments:
        env: RMAB Simulator environment
        lamb: \alpha, tradeoff between R_{i} and R_{glob}
        
    Returns: Q values, one for each combination of state + action"""

    match_probability = env.match_probability_list 
    if match_probability != []:
        match_probability = np.array(match_probability)[env.agent_idx]
    true_transitions = env.transitions
    discount = env.discount 
    budget = env.budget 

    Q_vals = arm_value_iteration_exponential(true_transitions,discount,budget,env.volunteers_per_arm,env.reward_type,env.reward_parameters,
                    reward_function=reward_function,lamb=lamb,
                    match_probability_list=match_probability)

    return Q_vals 

def q_iteration_custom_epoch():
    """Run Q Iteration with a custom reward function: 
    
    Arguments: None
    
    Returns: A policy which runs q_iteration using the custom reward function"""
    def q_iteration(env,lamb):
        return q_iteration_epoch(env,lamb,reward_function='custom')
    return q_iteration

def compute_p_matrix(env,N):
    """Compute a matrices of values of p_{i}(s) for all i, s
    Useful to compute Linear-Whittle indices
    
    Arguments:
        env: RMABSimulator Environment
        N: Integer, number of agents
    
    Returns: numpy matrix of size N x number of states"""

    n_states = env.transitions.shape[1]
    p_matrix = np.zeros((N,n_states))

    for i in range(N):
        for s in range(n_states):
            default_state = [env.worst_state for _ in range(N)]
            default_state[i] = s
            p_matrix[i,s] = custom_reward(default_state,one_hot(i,N),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.active_states)
    return p_matrix 

def compute_reward_matrix(env,N,lamb):
    """Compute the individual rewards for each arm
    We assume that this is \lambda/N for an active state
    Zero otherwise
    
    Arguments:
        env: RMABSimulator Environment
        N: Integer, number of agents
        lamb: Float, weight on the individual rewards vs. global reward
    
    Returns: 
        numpy array of size N x number of states
    """

    n_states = env.transitions.shape[1]
    reward_matrix = np.zeros((N,n_states,2))

    for i in range(N):
        for j in range(n_states):
            if j in env.active_states:
                reward_matrix[i,j] += lamb/N
    
    return reward_matrix 

