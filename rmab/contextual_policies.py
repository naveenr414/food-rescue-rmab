import numpy as np
import heapq

from rmab.uc_whittle import Memoizer
from rmab.compute_whittle import arm_compute_whittle, arm_value_iteration_exponential, arm_value_v_iteration, get_q_vals, fast_arm_compute_whittle_multi_prob
from itertools import combinations
from rmab.simulator import generate_random_context
from rmab.omniscient_policies import whittle_index

import torch
import torch.nn as nn
import torch.optim as optim
import itertools 
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from scipy.stats import binom
from copy import deepcopy
from mcts import mcts
import random 
import time
import scipy


def shapley_index_contextual(env,state,memoizer_shapley = {}):
    """Compute the Shapley index for matching; how much
        does match probability increase when using some particular arm
        
    Arguments:
        env: RMAB Simulator environment
        state: Numpy array of 0-1 states for each volunteer
        memoizer_shapley: Dictionary, to store previously computed Shapley indices
        
    Returns: Two things, shapley index, and updated dictionary"""
    shapley_indices = [0 for i in range(len(state))]
    state_str = "".join([str(i) for i in state])

    if state_str in memoizer_shapley:
        return memoizer_shapley[state_str], memoizer_shapley

    state_1 = [i for i in range(len(state)) if state[i] != 0]
    match_probabilities = np.array(env.current_episode_match_probs)[:,env.agent_idx]
    num_random_combos = 20*len(state_1)
    # num_random_combos = min(num_random_combos,100000)

    combinations = np.zeros((num_random_combos, len(match_probabilities)), dtype=int)

    budget = env.budget 

    # Fix for when the number of combinations is small (with respect to the budget)
    # In that scenario, we can essentially just manually compute
    budget_probs = np.array([scipy.special.comb(len(match_probabilities),k) for k in range(0,budget)])
    budget_probs /= np.sum(budget_probs)

    scores = []
    indices = []

    for i in range(num_random_combos):
        k = random.choices(list(range(len(budget_probs))), weights=budget_probs,k=1)[0]
        ones_indices = random.sample(list(range(len(match_probabilities))),k)
        combinations[i, ones_indices] = 1

        rand_index = random.randint(0,len(match_probabilities)//2)
        indices.append(rand_index)
        score = np.prod([(1-match_probabilities[rand_index,j]) for j in range(len(state)) if combinations[i,j]])
        scores.append(score)
    
    scores = np.array(scores)

    for i in range(len(state_1)):
        avg_shapley = 0
        for j in np.where(combinations[:, i] == 0)[0]:
            match_prob = match_probabilities[indices[j]][i]
            avg_shapley += scores[j] - (1-match_prob)*scores[j]
        avg_shapley /= len(np.where(combinations[:, i] == 0)[0])
        shapley_indices[i] = avg_shapley

    memoizer_shapley[state_str] = shapley_indices

    return shapley_indices, memoizer_shapley

def whittle_policy_contextual(env,state,budget,lamb,memory,per_epoch_results):
    """Whittle index policy based on computing the subsidy for each arm
    This approximates the problem as the sum of Linear rewards, then 
    Decomposes the problem into the problem for each arm individually
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Information on previously computed Whittle indices, the memoizer
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    N = len(state) 

    if memory == None:
        memoizer = Memoizer('optimal') 
        match_probabilities = env.current_episode_match_probs[:,env.agent_idx]
        match_probs = np.mean(match_probabilities,axis=0)
        #match_probs = [env.match_function(env.context,match_probabilities[i])*state[i] for i in range(N)]
    else:
        memoizer, match_probs = memory 

    state_WI = whittle_index(env,state,budget,lamb,memoizer,contextual=True,match_probs=match_probs)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (memoizer,match_probs) 

def whittle_policy_adjusted_contextual(env,state,budget,lamb,memory,per_epoch_results):
    """Whittle index policy based on computing the subsidy for each arm
    This approximates the problem as the sum of Linear rewards, then 
    Decomposes the problem into the problem for each arm individually
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Information on previously computed Whittle indices, the memoizer
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    N = len(state) 
    
    if memory == None:
        memoizer = Memoizer('optimal') 
        match_probabilities = env.current_episode_match_probs[:,env.agent_idx]
        match_probs = np.mean(match_probabilities,axis=0)
    else:
        memoizer, match_probs = memory 

    state_WI = whittle_index(env,state,budget,lamb,memoizer,contextual=True,match_probs=match_probs)
    activity_whittle = whittle_index(env,state,budget,lamb,memoizer,reward_function="activity")

    transitions = env.transitions[0]
    a = transitions[0,0,1]
    b = transitions[0,1,1]
    c = transitions[1,0,1]
    d = transitions[1,1,1]
    current_match_probs = env.current_episode_match_probs[env.timestep + env.episode_count*env.episode_len][env.agent_idx]
    state_WI += (current_match_probs-match_probs)*(1-lamb)

    pred_value = (d-c)*0.9/(1+0.9*(a-c))*lamb/len(state) + current_match_probs[2]*(1-lamb)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (memoizer,match_probs) 


def greedy_policy_contextual(env,state,budget,lamb,memory,per_epoch_results):
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

    match_probabilities = env.current_episode_match_probs[env.timestep + env.episode_count*env.episode_len][env.agent_idx]

    score_by_agent = [0 for i in range(N)]

    for i in range(N):        
        matching_score = state[i]*match_probabilities[i]
        score_by_agent[i] = matching_score

    selected_idx = np.argsort(score_by_agent)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1
    return action, memory

def whittle_greedy_contextual_policy(env,state,budget,lamb,memory, per_epoch_results):
    """Combination of the Whittle index + match probability
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and the Whittle memoizer"""


    N = len(state) 

    if memory == None:
        memoizer = Memoizer('optimal')
    else:
        memoizer = memory 

    state_WI = whittle_index(env,state,budget,lamb,memoizer,reward_function="activity")
    state_WI*=lamb 

    match_probabilities = env.current_episode_match_probs[env.timestep + env.episode_count*env.episode_len][env.agent_idx]
    match_probabilities *= state 

    state_WI += match_probabilities*(1-lamb)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, memoizer 

def shapley_whittle_contextual_policy(env,state,budget,lamb,memory, per_epoch_results):
    """Combination of the Whittle index + Shapley values
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and the Whittle memoizer"""


    N = len(state) 

    if memory == None:
        memory_whittle = Memoizer('optimal')
        memory_shapley = np.array(shapley_index_contextual(env,np.ones(len(state)),{})[0])
    else:
        memory_whittle, memory_shapley = memory 
        
    state_WI = whittle_index(env,state,budget,lamb,memory_whittle,reward_function="activity")
    state_WI*=lamb 

    state_WI += memory_shapley*(1-lamb)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (memory_whittle, memory_shapley)

def contextual_future_policy(env,state,budget,lamb,memory, per_epoch_results):
    """Select the policy which maximizes arm heterogenetiy
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""


    N = len(state)
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    
    values_by_combo = []

    for i in range(len(match_probabilities)):
        for j in range(i+1,len(match_probabilities)):
            prob_i = env.match_function(env.context,match_probabilities[i])*state[i] 
            prob_j = env.match_function(env.context,match_probabilities[j])*state[j]
            current_value = 1-(1-prob_i)*(1-prob_j)
            future_value = -1*np.abs(match_probabilities[i].dot(match_probabilities[j]))/(match_probabilities[i].dot(match_probabilities[i])*match_probabilities[j].dot(match_probabilities[j]))
            values_by_combo.append((current_value+future_value,i,j))

    values_by_combo = sorted(values_by_combo,key=lambda k: k[0])[::-1]
    i,j = values_by_combo[0][1], values_by_combo[0][2]
    action = np.zeros(N, dtype=np.int8)
    action[i] = 1
    action[j] = 1

    if budget > 2:
        all_values = [k for k in range(len(action)) if k not in [i,j]]
        random.shuffle(all_values)
        for k in all_values[:budget-2]:
            action[k] = 1

    return action, None
