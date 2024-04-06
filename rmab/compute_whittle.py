"""
standard Whittle index computation based on binary search

POSSIBLE OPTIMIZATIONS TO HELP SPEED
- keep track of top k WIs so far. then in future during binary search, if we go below that WI just quit immediately
"""

import numpy as np
from itertools import product, combinations
from rmab.utils import binary_to_decimal, list_to_binary, custom_reward
import random 

whittle_threshold = 1e-6
value_iteration_threshold = 1e-6

def get_q_vals(transitions, state, predicted_subsidy, discount, threshold=value_iteration_threshold,reward_function='activity',lamb=0,
                        match_prob=0.5,get_v=False,num_arms=1):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)
    """
    assert discount < 1

    n_states, n_actions = transitions.shape
    value_func = np.array([random.random() for i in range(n_states)])
    difference = np.ones((n_states))
    iters = 0


    # lambda-adjusted reward function
    def reward(s, a):
        return s/num_arms - a * predicted_subsidy

    def reward_matching(s,a):
        return s*a*match_prob - a*predicted_subsidy 

    def combined_reward(s,a):
        rew = s*a*match_prob*(1-lamb) + lamb*s/num_arms - a*predicted_subsidy 
        return rew 

    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        # calculate Q-function
        Q_func = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                if reward_function == 'activity':
                    r = reward  
                elif reward_function == 'matching':
                    r = reward_matching 
                elif reward_function == 'combined':
                    r = combined_reward
                else:
                    raise Exception("Reward function {} not found".format(reward_function))

                # transitioning to state = 0
                Q_func[s, a] += (1 - transitions[s, a]) * (r(s, a) + discount * value_func[0])

                # # transitioning to state = 1
                Q_func[s, a] += transitions[s, a] * (r(s, a) + discount * value_func[1])

            value_func[s] = np.max(Q_func[s, :])

        difference = np.abs(orig_value_func - value_func)

    if get_v:
        return Q_func[state,:], value_func

    # print(f'q values {Q_func[state, :]}, action {np.argmax(Q_func[state, :])}')
    return Q_func[state,:]

def arm_value_iteration_exponential(all_transitions, discount, budget, volunteers_per_arm, threshold=value_iteration_threshold,reward_function='matching',lamb=0,power=None,match_probability_list=[]):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)

    Arguments:
        all_transitions: N x num_states x num_actions (presumably 2) x num_states
        discount: Gamma, float

    Returns: Q_func, numpy matrix with Q values for each combination of states, 
        and each combination of actions
        This is encoded as a 2^N x 2^N matrix, where a state is encoded in binary
    """
    assert discount < 1
    n_arms, _ = all_transitions.shape[0], all_transitions.shape[2]
    
    N = len(match_probability_list)
    num_real_states = 2**(N)
    value_func = np.array([random.random() for i in range(num_real_states)])
    difference = np.ones((num_real_states))
    iters = 0
    match_probability_list = np.array(match_probability_list)
    
    all_s = np.array(list(product([0, 1], repeat=N)))
    all_s = [np.array(i) for i in all_s]
    
    all_a = []
    for b in range(budget+1):
        all_a += list(combinations(range(N), b))    

    all_a = [np.array(list_to_binary(i,N)) for i in all_a]

    def reward_activity(s,a):
        return np.sum(s)

    def reward_matching(s,a):
        return (1-np.prod(np.power(1-match_probability_list,s*a)))
        
    def reward_combined(s,a):
        rew = (1-np.prod(np.power(1-match_probability_list,s*a)))*(1-lamb) + lamb*np.sum(s)/len(s)
        return rew
    
    def reward_linear(s,a):
        rew = lamb*np.sum(s)/len(s) + (1-lamb)*np.sum(match_probability_list*s*a)
        return rew 

    def reward_submodular(s,a):
        # TODO: Change this back
        match_probs = match_probability_list*s*a
        return np.max(match_probs)*(1-lamb) + lamb*np.sum(s)/len(s)
        #return ((np.sum(match_probability_list*s*a)+1)**power-1)*(1-lamb) + lamb*np.sum(s)/len(s)

    def reward_custom(s,a):
        val = custom_reward(s,a,match_probability_list)*(1-lamb) + lamb*np.sum(s)/len(s)
        return val 

    if reward_function == 'activity':
        r = reward_activity
    elif reward_function == 'matching':
        r = reward_matching 
    elif reward_function == 'combined': 
        r = reward_combined 
    elif reward_function == 'submodular':
        assert power != None
        r = reward_submodular
    elif reward_function == 'custom': 
        r = reward_custom
    else:
        raise Exception("{} reward function not found".format(reward_function))

    # Precompute transition probabilities for speed 
    precomputed_transition_probabilities = np.zeros((num_real_states,num_real_states,num_real_states))
    
    for s in all_s:
        s_rep = binary_to_decimal(s) 

        for s_prime in all_s:
            s_prime_rep = binary_to_decimal(s_prime) 

            for a in all_a:
                a_rep = binary_to_decimal(a)
                transition_probability = np.prod([all_transitions[i//volunteers_per_arm][s[i]][a[i]][s_prime[i]]
                            for i in range(N)])
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

                for s_prime in all_s:
                    s_prime_rep = binary_to_decimal(s_prime)
                    Q_func[s_rep,a_rep] += precomputed_transition_probabilities[s_rep,a_rep,s_prime_rep] * (r(s,a)
                         + discount * value_func[s_prime_rep])
            value_func[s_rep] = np.max(Q_func[s_rep, :])
        difference = np.abs(orig_value_func - value_func)
    return Q_func 

def arm_value_iteration(transitions, state, predicted_subsidy, discount, threshold=value_iteration_threshold,reward_function='activity',lamb=0,
                        match_prob=0.5,num_arms=1):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)
    """
    return np.argmax(get_q_vals(transitions,state,predicted_subsidy,discount,threshold,reward_function=reward_function,lamb=lamb,match_prob=match_prob,num_arms=num_arms))

def arm_value_v_iteration(transitions, state, predicted_subsidy, discount, threshold=value_iteration_threshold,reward_function='activity',lamb=0,
                        match_prob=0.5):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)
    """

    Q_vals,v_vals = get_q_vals(transitions,state,predicted_subsidy,discount,threshold,reward_function=reward_function,lamb=lamb,match_prob=match_prob,get_v=True)    
    return np.argmax(Q_vals), np.max(Q_vals), v_vals 

def get_init_bounds(transitions,lamb=0):
    lamb = max(lamb,1)+1
    lb = -lamb
    ub = lamb
    return lb, ub

def arm_compute_whittle(transitions, state, discount, subsidy_break, eps=whittle_threshold,reward_function='activity',lamb=0,match_prob=0.5,match_probability_list=[],get_v=False,num_arms=1):
    """
    compute whittle index for a single arm using binary search

    subsidy_break = the min value at which we stop iterating

    param transitions:
    param eps: epsilon convergence
    returns Whittle index
    """
    lb, ub = get_init_bounds(transitions,lamb) # return lower and upper bounds on WI

    while abs(ub - lb) > eps:
        predicted_subsidy = (lb + ub) / 2

        # we've already filled our knapsack with higher-valued WIs
        if ub < subsidy_break:
            return -10

        if get_v:
            action, value, v_val = arm_value_v_iteration(transitions, state, predicted_subsidy, discount,reward_function=reward_function,lamb=lamb,
                        match_prob=match_prob)
        else:
            action = arm_value_iteration(transitions, state, predicted_subsidy, discount,reward_function=reward_function,lamb=lamb,
                        match_prob=match_prob,num_arms=num_arms)

        if action == 0:
            # optimal action is passive: subsidy is too high
            ub = predicted_subsidy
        elif action == 1:
            # optimal action is active: subsidy is too low
            lb = predicted_subsidy
        else:
            raise Exception(f'action not binary: {action}')
    
    subsidy = (ub + lb) / 2

    if get_v:
        return subsidy, value, v_val

    return subsidy
