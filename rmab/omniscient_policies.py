""" Oracle algorithms for matching, activity """

import numpy as np
import heapq

from rmab.uc_whittle import Memoizer
from rmab.compute_whittle import arm_compute_whittle, arm_value_iteration_exponential, arm_value_iteration_neural, arm_compute_whittle_sufficient, arm_value_v_iteration, get_q_vals
from rmab.utils import get_stationary_distribution, binary_to_decimal, list_to_binary
from itertools import combinations

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

def whittle_index(env,state,budget,lamb,memoizer,reward_function="combined",shapley_values=None):
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
    min_chosen_subsidy = -1 
    for i in range(N):
        arm_transitions = true_transitions[i//env.volunteers_per_arm, :, :, 1]
        check_set_val = memoizer.check_set(arm_transitions+round(match_probability_list[i],2), state[i])
        if check_set_val != -1:
            state_WI[i] = check_set_val
        else:
            state_WI[i] = arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=match_probability_list[i])
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
                        match_prob=match_probability_list[i]) #arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=match_probability_list[i],get_v=True)
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
    num_random_combos = min(num_random_combos,100000)

    combinations = np.zeros((num_random_combos, len(corresponding_probabilities)), dtype=int)

    budget = env.budget 

    # TODO: Fix this later
    if len(corresponding_probabilities) <= env.budget-1:
        if len(corresponding_probabilities) == 1:
            return match_probabilities * state, memoizer_shapley 
        budget = 2

    for i in range(num_random_combos):
        ones_indices = np.random.choice(len(corresponding_probabilities),budget-1, replace=False)
        combinations[i, ones_indices] = 1

    scores = [np.prod(1-corresponding_probabilities[combo == 1]) for combo in combinations]
    scores = np.array(scores)

    for i in range(len(state_1)):
        shapley_indices[state_1[i]] = np.mean(scores[combinations[:,i] == 0])-np.mean(scores[combinations[:,i] == 0]*(1-match_probabilities[i]))

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
        print("For {} indices are {}".format(trial,indices))

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

    Q_vals = per_epoch_results

    N = len(state)

    state_rep = binary_to_decimal(state)

    for i in range(2**4):
        s = bin(i)[2:].zfill(N)
        best_action = np.argmax(Q_vals[i])
        binary_val = bin(best_action)[2:].zfill(N)
        print("{} {}".format(s,binary_val))
    z = 1/0 

    max_action = np.argmax(Q_vals[state_rep])
    binary_val = bin(max_action)[2:].zfill(N)

    action = np.zeros(N, dtype=np.int8)
    action = np.array([int(i) for i in binary_val])

    return action, None

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

    # select arms at random
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

    state_WI = whittle_index(env,state,budget,lamb,memoizer)
    state_WI*=lamb 

    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]*state
    match_probabilities /= (1-env.discount)

    state_WI += match_probabilities

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
        memoizer = Memoizer('optimal')
        memoizer_shapley = {}
    else:
        memoizer, memoizer_shapley = memory 
        

    state_WI = whittle_index(env,state,budget,lamb,memoizer)
    state_WI*=lamb 

    shapley_indices, memoizer_shapley = shapley_index(env,state,memoizer_shapley)

    state_WI += shapley_indices

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (memoizer, memoizer_shapley)



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



def run_heterogenous_policy(env, n_episodes, n_epochs,discount,policy,seed,per_epoch_function=None,lamb=0):
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

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    start = time.time()
    for epoch in range(n_epochs):
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

            action,memory = policy(env,state,budget,lamb,memory,per_epoch_results)
            next_state, reward, done, _ = env.step(action)


            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward
    print("Took {} time".format(time.time()-start))
    env.time_taken = time.time()-start

    activity_rate = env.total_active/(all_reward.size*env.cohort_size*env.volunteers_per_arm)

    return all_reward, activity_rate


