"""
standard Whittle index computation based on binary search

POSSIBLE OPTIMIZATIONS TO HELP SPEED
- keep track of top k WIs so far. then in future during binary search, if we go below that WI just quit immediately
"""

import sys
import numpy as np
import scipy.misc
from itertools import product, combinations
import time 


import heapq  # priority queue

whittle_threshold = 1e-4
value_iteration_threshold = 1e-2

def arm_value_iteration(transitions, state, lamb_val, discount, threshold=value_iteration_threshold,use_match_reward=False):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)
    """
    assert discount < 1

    n_states, n_actions = transitions.shape
    value_func = np.random.rand(n_states)
    difference = np.ones((n_states))
    iters = 0

    # lambda-adjusted reward function
    def reward(s, a):
        return s - a * lamb_val

    def reward_matching(s,a):
        return s*a - a*lamb_val 

    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        # calculate Q-function
        Q_func = np.zeros((n_states, n_actions))
        for s in range(n_states):
            Q_val_s0 = 0
            Q_val_s1 = 0
            for a in range(n_actions):
                r = reward 

                if use_match_reward:
                    r = reward_matching

                # transitioning to state = 0
                Q_func[s, a] += (1 - transitions[s, a]) * (r(s, a) + discount * value_func[0])

                # transitioning to state = 1
                Q_func[s, a] += transitions[s, a] * (r(s, a) + discount * value_func[1])

            value_func[s] = np.max(Q_func[s, :])

        difference = np.abs(orig_value_func - value_func)

    # print(f'q values {Q_func[state, :]}, action {np.argmax(Q_func[state, :])}')
    return np.argmax(Q_func[state, :])


def get_init_bounds(transitions):
    lb = -1
    ub = 1
    return lb, ub


def arm_compute_whittle(transitions, state, discount, subsidy_break, eps=whittle_threshold,use_match_reward=False):
    """
    compute whittle index for a single arm using binary search

    subsidy_break = the min value at which we stop iterating

    param transitions:
    param eps: epsilon convergence
    returns Whittle index
    """
    lb, ub = get_init_bounds(transitions) # return lower and upper bounds on WI
    top_WI = []
    while abs(ub - lb) > eps:
        lamb_val = (lb + ub) / 2
        # print('lamb', lamb_val, lb, ub)

        # we've already filled our knapsack with higher-valued WIs
        if ub < subsidy_break:
            # print('breaking early!', subsidy_break, lb, ub)
            return -10

        action = arm_value_iteration(transitions, state, lamb_val, discount,use_match_reward=use_match_reward)
        if action == 0:
            # optimal action is passive: subsidy is too high
            ub = lamb_val
        elif action == 1:
            # optimal action is active: subsidy is too low
            lb = lamb_val
        else:
            raise Error(f'action not binary: {action}')
    subsidy = (ub + lb) / 2
    return subsidy

def binary_to_decimal(binary_list):
    """Turn 0-1 lists into a number, for state representation
    
    Arguments:
        binary_list: List of 0,1
    
    Returns: Integer base-10 represnetation"""

    decimal_value = 0
    for bit in binary_list:
        decimal_value = decimal_value * 2 + bit
    return decimal_value

def list_to_binary(a,n_arms):
    """Given a list of the form [0,3,5], return a binary
        array of length n_arms with 1 if i is in a, 0 otherwise
        For example, [1,0,0,1,0,1]
    
    Arguments: a, numpy array or list
        n_arms: Integer, length of the return list
    
    Returns: 0-1 List of length n_arms"""

    return np.array([1 if i in a else 0 for i in range(n_arms)])

def arm_value_iteration_match(all_transitions, discount, budget, match_prob, threshold=value_iteration_threshold,lamb=0):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)

    Arguments:
        all_transitions: N x num_states x num_actions (presumably 2) x num_states
        discount: Gamma, float
        match_prob: Float, match probability for each arm

    Returns: Q_func, numpy matrix with Q values for each combination of states, 
        and each combination of actions
        This is encoded as a 2^N x 2^N matrix, where a state is encoded in binary
    """
    assert discount < 1
    n_arms, _ = all_transitions.shape[0], all_transitions.shape[2]
    num_real_states = 2**(n_arms)
    value_func = np.random.rand(num_real_states)
    difference = np.ones((num_real_states))
    iters = 0
    p = match_prob 
    
    all_s = np.array(list(product([0, 1], repeat=n_arms)))
    all_s = [np.array(i) for i in all_s]
    
    all_a = list(combinations(range(n_arms), budget))
    all_a = [np.array(list_to_binary(i,n_arms)) for i in all_a]

    def reward_matching(s,a):
        return (1-np.power(1-p,s.dot(a))) + lamb*np.sum(s)

    # Precompute transition probabilities for speed 
    precomputed_transition_probabilities = np.zeros((num_real_states,num_real_states,num_real_states))
    for s in all_s:
        s_rep = binary_to_decimal(s) 

        for s_prime in all_s:
            s_prime_rep = binary_to_decimal(s_prime) 

            for a in all_a:
                a_rep = binary_to_decimal(a)
                transition_probability = np.prod([all_transitions[i][s[i]][a[i]][s_prime[i]]
                            for i in range(n_arms)])
                precomputed_transition_probabilities[s_rep][a_rep][s_prime_rep] = transition_probability
    

    # Perform Q Iteration 
    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        Q_func = np.zeros((num_real_states, num_real_states))
        for s in all_s:
            s_rep = binary_to_decimal(s) 

            for a in all_a:
                a_rep = binary_to_decimal(a)
                action = np.zeros(n_arms)
                for i in a:
                    action[i] = 1 

                for s_prime in all_s:
                    s_prime_rep = binary_to_decimal(s_prime)
                    Q_func[s_rep,a_rep] += precomputed_transition_probabilities[s_rep,a_rep,s_prime_rep] * (reward_matching(s,a)
                         + discount * value_func[s_prime_rep])
            value_func[s_rep] = np.max(Q_func[s_rep, :])
        difference = np.abs(orig_value_func - value_func)

    return Q_func 
