import numpy as np
from rmab.utils import custom_reward, one_hot

import random 

def compute_p_matrix(env,N):
    n_states = env.transitions.shape[1]
    p_matrix = np.zeros((N,n_states))

    for i in range(N):
        for s in range(n_states):
            default_state = [env.worst_state for _ in range(N)]
            default_state[i] = s
            p_matrix[i,s] = custom_reward(default_state,one_hot(i,N),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters)
    return p_matrix 


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
        memory = compute_p_matrix(env,N)

    score_by_agent = [memory[i][state[i]] for i in range(N)]
    selected_idx = np.argsort(score_by_agent)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, memory 

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