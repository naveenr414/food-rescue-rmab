""" Oracle algorithms for matching, activity """

import numpy as np
import heapq

from rmab.uc_whittle import Memoizer
from rmab.compute_whittle import arm_compute_whittle, arm_value_iteration_exponential, arm_value_iteration_neural, arm_compute_whittle_sufficient, arm_value_v_iteration, get_q_vals
from rmab.utils import get_stationary_distribution, binary_to_decimal, list_to_binary
from itertools import combinations
from rmab.simulator import generate_random_context

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

def whittle_index(env,state,budget,lamb,memoizer,reward_function="combined",shapley_values=None,contextual=False,match_probs=None):
    """Get the Whittle indices for each agent in a given state
    
    Arguments:
        env: Simualtor RMAB environment
        state: Numpy array of 0-1 for each volunteer
        budget: Max arms to pick 
        lamb: Float, balancing matching and activity
        memoizer: Object that stores previous Whittle index computations
    
    Returns: List of Whittle indices for each arm"""
    N = len(state) 
    match_probability_list = np.array(env.match_probability_list)[env.agent_idx]

    if shapley_values != None:
        match_probability_list = np.array(shapley_values)

    if contextual:
        match_probability_list = match_probs

    true_transitions = env.transitions 
    discount = env.discount 

    state_WI = np.zeros((N))
    min_chosen_subsidy = -1 
    for i in range(N):
        arm_transitions = true_transitions[i//env.volunteers_per_arm, :, :, 1]
        if reward_function == "activity":
            check_set_val = memoizer.check_set(arm_transitions, state[i])
        else:
            check_set_val = memoizer.check_set(arm_transitions+round(match_probability_list[i],2), state[i])
        if check_set_val != -1:
            state_WI[i] = check_set_val
        else:
            state_WI[i] = arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=match_probability_list[i],num_arms=len(state))
            if reward_function == "activity":
                memoizer.add_set(arm_transitions, state[i], state_WI[i])
            else:
                memoizer.add_set(arm_transitions+round(match_probability_list[i],2), state[i], state_WI[i])
    
            

    return state_WI


def whittle_v_index(env,state,budget,lamb,memoizer,reward_function="combined",shapley_values=None):
    """Get the Whittle indices for each agent in a given state
    
    Arguments:
        env: Simualtor RMAB environment
        state: Numpy array of 0-1 for each volunteer
        budget: Max arms to pick 
        lamb: Float, balancing matching and activity
        memoizer: Object that stores previous Whittle index computations
    
    Returns: List of Whittle indices for each arm"""
    N = len(state) 
    match_probability_list = np.array(env.match_probability_list)[env.agent_idx]

    if shapley_values != None:
        match_probability_list = np.array(shapley_values)

    true_transitions = env.transitions 
    discount = env.discount 

    state_WI = np.zeros((N))
    state_V = np.zeros((N))
    state_V_full = np.zeros((N,2))

    min_chosen_subsidy = -1 
    for i in range(N):
        arm_transitions = true_transitions[i//env.volunteers_per_arm, :, :, 1]
        check_set_val = memoizer.check_set(arm_transitions+round(match_probability_list[i],2), state[i])
        if check_set_val != -1:
            state_WI[i], state_V[i], state_V_full[i] = check_set_val
        else:
            state_WI[i], state_V[i], state_V_full[i] = arm_value_v_iteration(arm_transitions, state, 0, discount,reward_function=reward_function,lamb=lamb,
                        match_prob=match_probability_list[i]) 
            memoizer.add_set(arm_transitions+round(match_probability_list[i],2), state[i], (state_WI[i],state_V[i],state_V_full[i]))

    return state_WI, state_V, state_V_full


def shapley_index(env,state,memoizer_shapley = {}):
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
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    corresponding_probabilities = match_probabilities[state_1]
    num_random_combos = 20*len(state_1)
    # num_random_combos = min(num_random_combos,100000)

    combinations = np.zeros((num_random_combos, len(corresponding_probabilities)), dtype=int)

    budget = env.budget 

    # Fix for when the number of combinations is small (with respect to the budget)
    # In that scenario, we can essentially just manually compute
    if len(corresponding_probabilities) <= env.budget-1:
        if len(corresponding_probabilities) == 1:
            return match_probabilities * state, memoizer_shapley
        else: 
            budget = 2

    budget_probs = np.array([scipy.special.comb(len(corresponding_probabilities),k) for k in range(0,budget)])
    budget_probs /= np.sum(budget_probs)

    for i in range(num_random_combos):
        k = np.random.choice(len(budget_probs), p=budget_probs)
        ones_indices = np.random.choice(len(corresponding_probabilities),k, replace=False)
        combinations[i, ones_indices] = 1

    scores = [np.prod(1-corresponding_probabilities[combo == 1]) for combo in combinations]
    scores = np.array(scores)

    for i in range(len(state_1)):
        shapley_indices[state_1[i]] = np.mean(scores[combinations[:,i] == 0])-np.mean(scores[combinations[:,i] == 0]*(1-match_probabilities[i]))

    memoizer_shapley[state_str] = shapley_indices

    return shapley_indices, memoizer_shapley

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
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    corresponding_probabilities = match_probabilities[state_1]
    num_random_combos = 20*len(state_1)
    # num_random_combos = min(num_random_combos,100000)

    combinations = np.zeros((num_random_combos, len(corresponding_probabilities)), dtype=int)
    random_contexts = np.array([generate_random_context(env.context_dim) for i in range(num_random_combos)])

    budget = env.budget 

    # Fix for when the number of combinations is small (with respect to the budget)
    # In that scenario, we can essentially just manually compute
    if len(corresponding_probabilities) <= env.budget-1:
        if len(corresponding_probabilities) == 1:
            return match_probabilities * state, memoizer_shapley
        else: 
            budget = 2

    budget_probs = np.array([scipy.special.comb(len(corresponding_probabilities),k) for k in range(0,budget)])
    budget_probs /= np.sum(budget_probs)

    scores = []

    for i in range(num_random_combos):
        k = np.random.choice(len(budget_probs), p=budget_probs)
        ones_indices = np.random.choice(len(corresponding_probabilities),k, replace=False)
        combinations[i, ones_indices] = 1
        score = np.prod([(1-env.match_function(random_contexts[i],match_probabilities[j])) for j in range(len(state)) if combinations[i,j]])
        scores.append(score)
    
    scores = np.array(scores)

    for i in range(len(state_1)):
        avg_shapley = 0
        for j in np.where(combinations[:, i] == 0)[0]:
            match_prob = env.match_function(random_contexts[j],match_probabilities[i])
            avg_shapley += scores[j] - (1-match_prob)*scores[j]
        avg_shapley /= len(np.where(combinations[:, i] == 0)[0])
        shapley_indices[i] = avg_shapley

    memoizer_shapley[state_str] = shapley_indices

    return shapley_indices, memoizer_shapley

def whittle_activity_policy(env,state,budget,lamb,memory,per_epoch_results):
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
    else:
        memoizer = memory 

    state_WI = whittle_index(env,state,budget,lamb,memoizer,reward_function="activity")

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, memoizer 

def whittle_policy(env,state,budget,lamb,memory,per_epoch_results):
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
    else:
        memoizer = memory 

    state_WI = whittle_index(env,state,budget,lamb,memoizer)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, memoizer 

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
        match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
        match_probs = [env.get_average_prob(match_probabilities[i],100) for i in range(N)]
    else:
        memoizer, match_probs = memory 

    state_WI = whittle_index(env,state,budget,lamb,memoizer,contextual=True,match_probs=match_probs)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (memoizer,match_probs) 


def whittle_whittle_policy(env,state,budget,lamb,memory,per_epoch_results):
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
        memoizer = [Memoizer('optimal'),Memoizer('optimal'),shapley_index(env,np.ones(len(state)),{})]
    else:
        memoizer = memory 
    
    state_WI_activity = whittle_index(env,state,budget,lamb,memoizer[0],reward_function="activity")
    state_WI_matching = whittle_index(env,state,budget,lamb,memoizer[1],reward_function="combined",shapley_values=memoizer[2][0])

    combined_WI = state_WI_matching+lamb*state_WI_activity

    sorted_WI = np.argsort(combined_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, memoizer 
 

def q_iteration_epoch(env,lamb):
    """Compute Q Values for all combinations of agents in a given environment
    
    Arguments:
        env: RMAB Simulator environment
        
    Returns: Q values, one for each combination of state + action"""

    match_probability = env.match_probability_list 
    if match_probability != []:
        match_probability = np.array(match_probability)[env.agent_idx]
    true_transitions = env.transitions
    discount = env.discount 
    budget = env.budget 

    Q_vals = arm_value_iteration_exponential(true_transitions,discount,budget,env.volunteers_per_arm,
                    reward_function='combined',lamb=lamb,
                    match_probability_list=match_probability)

    return Q_vals 

def index_computation_policy(env,state,budget,lamb,memory,per_epoch_results):
    """Q Iteration policy that computes Q values for all combinations of states
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: The Q Values
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    Q_vals = per_epoch_results
    N = len(state)

    indices = np.zeros(N)
    state_rep = binary_to_decimal(state)

    for trial in range(5):
        for i in range(N):
            max_index = 10
            min_index = 0

            for _ in range(20):
                predicted_index = (max_index+min_index)/2 
                other_agents = [i_prime for i_prime in range(N) if indices[i_prime]>=predicted_index and i_prime != i]
                agent_vals = np.array(env.match_probability_list)[env.agent_idx]*state

                other_agents = sorted(other_agents,key=lambda k: agent_vals[k],reverse=True)

                agents_with_i = set(other_agents[:budget-1] + [i])
                binary_with_i = binary_to_decimal([1 if i in agents_with_i else 0 for i in range(N)])
                agents_without_i = set(other_agents[:budget-1])
                binary_without_i = binary_to_decimal([1 if i in agents_without_i else 0 for i in range(N)])

                q_with_i = Q_vals[state_rep,binary_with_i]
                q_without_i = Q_vals[state_rep,binary_without_i]

                if q_with_i > q_without_i + predicted_index:
                    min_index = predicted_index 
                else:
                    max_index = predicted_index 
            indices[i] = (max_index+min_index)/2

    indices = np.argsort(indices)[-budget:][::-1]

    action = np.zeros(N, dtype=np.int8)
    action[indices] = 1
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
            print("In state {} best action is {} {}".format(state,action,value))

    max_action = np.argmax(Q_vals[state_rep])
    binary_val = bin(max_action)[2:].zfill(N)

    action = np.zeros(N, dtype=np.int8)
    action = np.array([int(i) for i in binary_val])

    return action, None

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

    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    match_probabilities = [env.match_function(env.context,match_probabilities[i]) for i in range(N)]

    score_by_agent = [0 for i in range(N)]

    for i in range(N):        
        matching_score = state[i]*match_probabilities[i]
        score_by_agent[i] = matching_score

    selected_idx = np.argsort(score_by_agent)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, memory

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
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]

    for i in range(N):
        activity_score = np.sum(state)
        
        matching_score = state[i]*match_probabilities[i]
        score_by_agent[i] = matching_score + activity_score * lamb 

    selected_idx = np.argsort(score_by_agent)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, None

def greedy_one_step_policy(env,state,budget,lamb,memory,per_epoch_results):
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
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    transitions = env.transitions

    for i in range(N):
        activity_score = transitions[i//env.volunteers_per_arm][state[i]][1][1]
        
        matching_score = state[i]*match_probabilities[i]
        score_by_agent[i] = matching_score + activity_score * lamb 

    # select arms at random
    selected_idx = np.argsort(score_by_agent)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, None

def whittle_iterative(env,state,budget,lamb,memory, per_epoch_results):
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

    true_transitions = env.transitions

    if memory == None:
        memoizer = Memoizer('optimal')
    else:
        memoizer = memory 

    # TODO: Make this more general than \lamb = 0
    # Compute this for \lamb = 0 for now 

    people_to_add = set()

    if memory == None:
        memoizer = [Memoizer('optimal'),Memoizer('optimal')]
    else:
        memoizer = memory 
    
    # state_WI_activity, state_V_activity = whittle_v_index(env,state,budget,lamb,memoizer[0],reward_function="activity")
    state_WI_matching, state_V_matching, state_V_full_matching = whittle_v_index(env,state,budget,lamb,memoizer[1],reward_function="matching")

    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]

    true_transitions = env.transitions 
    max_probabilities = true_transitions[:,1,1,1]
    probable_future_value = 1

    for _ in range(budget):
        values = [0 for j in range(N)]
        previous_val = 1-np.prod([1-match_probabilities[j]*state[j] for j in list(people_to_add)])

        for i in range(N):
            if i not in people_to_add:
                current_val = match_probabilities[i]*state[i]
                # future_val = state_V_matching[i] + state_WI_matching[i]*max_probabilities[i]*(1/(1-env.discount)-1)
                # future_val -= current_val 
                # future_val *= env.discount
                #future_val = (state_WI_matching[i] - match_probabilities[i])/env.discount 
                future_val = state_V_full_matching[i,0]*true_transitions[i//env.volunteers_per_arm,state[i],1,0] + state_V_full_matching[i][1]*true_transitions[i//env.volunteers_per_arm,state[i],1,1]
                future_val -=  state_V_full_matching[i,0]*true_transitions[i//env.volunteers_per_arm,state[i],0,0] + state_V_full_matching[i][1]*true_transitions[i//env.volunteers_per_arm,state[i],0,1]
                future_val *= env.discount 

                future_match_prob = match_probabilities[i]*true_transitions[i//env.volunteers_per_arm,state[i],1,1]

                real_current_val = 1-np.prod([1-match_probabilities[j]*state[j] for j in list(people_to_add)])*(1-match_probabilities[i])
                ratio = (real_current_val - previous_val)/(match_probabilities[i]*state[i])
                ratio_future = (1-probable_future_value*(1-future_match_prob) - (1-probable_future_value))/(future_match_prob)
                if match_probabilities[i]*state[i] == 0:
                    ratio = 0
                if future_match_prob == 0:
                    ratio_future = 0 
                total_val = future_val*ratio_future + current_val*ratio  
                #total_val = future_val*ratio_future + current_val*ratio 
                values[i] = total_val 
        
        if np.max(values) > 0:
            idx = np.argmax(values)
            people_to_add.add(idx)
            probable_future_value *= (1-match_probabilities[idx]*true_transitions[
                idx//env.volunteers_per_arm,state[idx],1,1])
        else:
            break 

    people_to_add = list(people_to_add)

    action = np.zeros(N, dtype=np.int8)
    action[people_to_add] = 1

    return action, memoizer 



def whittle_greedy_policy(env,state,budget,lamb,memory, per_epoch_results):
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

    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]*state

    state_WI += (1-lamb)*match_probabilities

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, memoizer 

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

    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    match_probabilities = np.array([env.match_function(env.context,i) for i in match_probabilities])
    match_probabilities *= state

    state_WI += match_probabilities*(1-lamb)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, memoizer 


def shapley_whittle_policy(env,state,budget,lamb,memory, per_epoch_results):
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
        memory_shapley = np.array(shapley_index(env,np.ones(len(state)),{})[0])
    else:
        memory_whittle, memory_shapley = memory 
        
    state_WI = whittle_index(env,state,budget,lamb,memory_whittle,reward_function="activity")
    state_WI*=lamb 

    state_WI += memory_shapley*(1-lamb)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (memory_whittle, memory_shapley)

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
    selected_idx = np.random.choice(N, size=budget, replace=False)
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, None

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
            future_value = -1*np.abs(match_probabilities[i].dot(match_probabilities[j]))
            values_by_combo.append((current_value+future_value,i,j))

    values_by_combo = sorted(values_by_combo,key=lambda k: k[0])[::-1]
    i,j = values_by_combo[0][1], values_by_combo[0][2]
    action = np.zeros(N, dtype=np.int8)
    action[i] = 1
    action[j] = 1

    return action, None

def run_heterogenous_policy(env, n_episodes, n_epochs,discount,policy,seed,per_epoch_function=None,lamb=0,get_memory=False,should_train=False,test_T=0):
    """Wrapper to run policies without needing to go through boilerplate code
    
    Arguments:
        env: Simualtor environment
        n_episodes: How many episodes to run for each epoch
            T = n_episodes * episode_len
        n_epochs: Number of different epochs/cohorts to run
        discount: Float, how much to discount rewards by
        policy: Function that takes in environment, state, budget, and lambda
            produces action as a result
        seed: Random seed for run
        lamb: Float, tradeoff between matching, activity
    
    Returns: Two things
        matching reward - Numpy array of Epochs x T, with rewards for each combo
        activity rate - Average rate of engagement across all volunteers
        We aim to maximize matching reward + lamb*n_arms*activity_rate"""

    N         = env.cohort_size*env.volunteers_per_arm
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env.reset_all()

    if should_train:
        all_reward = np.zeros((n_epochs,test_T))
        all_active_rate = np.zeros((n_epochs,test_T))
    else:
        all_reward = np.zeros((n_epochs, T))
        all_active_rate = np.zeros((n_epochs,T))


    inference_time_taken = 0

    for epoch in range(n_epochs):
        if not should_train:
            start = time.time()

        if epoch != 0: env.reset_instance()
        first_state = env.observe()
        if len(first_state)>20:
            first_state = first_state[:20]

        if per_epoch_function:
            per_epoch_results = per_epoch_function(env,lamb)
        else:
            per_epoch_results = None 

        memory = None 
        for t in range(0, T):
            state = env.observe()
            if should_train:
                if t>=T-test_T:
                    all_active_rate[epoch,t-(T-test_T)] = np.sum(state)/len(state)
            else:
                all_active_rate[epoch,t] = np.sum(state)/len(state)

            action,memory = policy(env,state,budget,lamb,memory,per_epoch_results)
            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            if should_train:
                if t == T-test_T:
                    start = time.time()

                if t < (T-test_T):
                    env.total_active = 0
                else:
                    all_reward[epoch, t-(T-test_T)] = reward
            else:
                all_reward[epoch, t] = reward
        inference_time_taken += time.time()-start 
    print("Took {} time".format(inference_time_taken))
    env.time_taken = inference_time_taken

    if get_memory:
        return all_reward, all_active_rate, memory
    return all_reward, all_active_rate

def get_discounted_reward(global_reward,active_rate,discount,lamb):
    """Compute the discounted combination of global reward and active rate
    
    Arguments: 
        global_reward: numpy array of size n_epochs x T
        active_rate: numpy array of size n_epochs x T
        discount: float, gamma

    Returns: Float, average discounted reward across all epochs"""

    all_rewards = []
    combined_reward = global_reward*(1-lamb) + lamb*active_rate

    for epoch in range(len(global_reward)):
        reward = 0
        for t in range(len(global_reward[epoch])):
            reward += combined_reward[epoch,t]*discount**(t)
        all_rewards.append(reward)
    
    return np.mean(all_rewards)