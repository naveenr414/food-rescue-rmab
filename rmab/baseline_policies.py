import numpy as np
from rmab.utils import custom_reward, one_hot

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

    score_by_agent = [0 for i in range(N)]
    match_probabilities = [custom_reward(one_hot(i,len(state)),one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters) for i in range(len(state))]
    for i in range(N):
        activity_score = np.sum(state)
        
        matching_score = state[i]*match_probabilities[i]
        score_by_agent[i] = matching_score + activity_score * lamb 

    selected_idx = np.argsort(score_by_agent)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, None

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