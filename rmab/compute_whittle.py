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

def arm_value_iteration_exponential(all_transitions, discount, budget, volunteers_per_arm, reward_type,reward_parameters,threshold=value_iteration_threshold,reward_function='matching',lamb=0,power=None,match_probability_list=[]):
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
        val = custom_reward(s,a,match_probability_list,reward_type,reward_parameters)*(1-lamb) + lamb*np.sum(s)/len(s)
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

def get_multi_Q(state,action,env,lamb,match_prob,match_prob_now):
    """Compute the total reward (in terms of the Q value) given an action
        And a set of match probabilities now/later
        
    Arguments: 
        state: Numpy array list of 0-1
        action: Numpy array list of 0-1
        env: RMAB Simulator
        lamb: Float, \lambda, tradeoff between engagement, global reward
        match_prob: List of marginal global rewards
        match_prob_now: List of marginal global rewards in the current time step
        
    Returns: Float, total Q value of playing action, given the environment structure"""

    Q_values = []
    for i in range(len(state)):
        if state[i] == 1:
            Q_fast_predicted = fast_Q_multi_prob(env.transitions[i//env.volunteers_per_arm,:,:,1], state[i], env.discount,lamb=lamb,
            match_prob=match_prob[i],match_prob_now=match_prob_now[i],num_arms=len(state))
            Q_values.append(Q_fast_predicted[action[i]])
        else:
            Q_fast_predicted = fast_Q(env.transitions[i//env.volunteers_per_arm,:,:,1], state[i], env.discount,lamb=lamb,
            match_prob=match_prob[i],num_arms=len(state))
            Q_values.append(Q_fast_predicted[action[i]])

    return np.sum(Q_values)
        
def fast_Q(transitions, state, discount,lamb=0.5,match_prob=0.5,num_arms=1):
    """Compute Q values by explicitly solving for the Q values (using Linear Algebra), rather  than running Value Iteration
    
    Arguments: 
        transitions: 2x2 numpy array of transitions to the 1 state
        state: 0-1 integer, which state one arm is in
        predicted_subsidy: How much to penalize pulling an arm by
        discount: How much to discount future rewards by, \gamma
        lamb: How much to balance engagement vs. global reward
        match_prob: Marginal reward to the global reward for pulling an arm
        num_arms: Total numbers of arms
        
    Returns: Numpy array of the Q values"""
    
    a = transitions[0,0]
    b = transitions[0,1]
    c = transitions[1,0]
    d = transitions[1,1]
    g = discount 
    q = match_prob 
    N = num_arms 

    if state == 0:
        Q_val_0 = -((g*(a*(-1 + g) - b*g)*(-lamb + (-1 + lamb)*N*q))/((-1 + g)*(1 + b*g - d*g)*N))
        Q_val_1 = (b*g*(-lamb + (-1 + lamb)*N*q))/((-1 + g)*(1 + b*g - d*g)*N)
    else:
        Q_val_0 = (lamb*(-1 + g*(1 - b + d + c (-1 + g) - d*g)) + (-1 + lamb)*g (c + b*g - c*g)*N*q)/((-1 + g)*(1 + b*g - d*g)*N)
        Q_val_1 = ((1 + (-1 + b)*g)*(-lamb + (-1 + lamb)*N*q))/((-1 + g)*(1 + b*g - d*g)*N)

    return np.array([Q_val_0,Q_val_1])

def fast_Q_multi_prob(transitions, state, discount, lamb=0.5,match_prob=0.5,match_prob_now=1,num_arms=1):
    """Compute Q values by explicitly solving for the Q values (using Linear Algebra), rather  than running Value Iteration
    
    Arguments: 
        transitions: 2x2 numpy array of transitions to the 1 state
        state: 0-1 integer, which state one arm is in
        predicted_subsidy: How much to penalize pulling an arm by
        discount: How much to discount future rewards by, \gamma
        lamb: How much to balance engagement vs. global reward
        match_prob: Marginal reward to the global reward for pulling an arm
        match_prob_now: Marginal reward in the current state for pulling an arm
        num_arms: Total numbers of arms
        
    Returns: Numpy array of the Q values"""

    a = transitions[0,0]
    b = transitions[0,1]
    c = transitions[1,0]
    d = transitions[1,1]
    g = discount 
    p = match_prob_now 
    q = match_prob 
    N = num_arms 
    assert state == 1

    Q_val_0 = (lamb*(-1 + g*(1 - b + d + c*(-1 + g) - d*g)) + (-1 + lamb)*g*(c + b*g - c*g)*N*q)/((-1 + g)*(1 + b*g - d*g)*N)
    Q_val_1 = (1/((-1 + g)*(1 + b*g - d*g)*N))*((-1 + g)*(1 + b*g - d*g)*N*p - g*(d + b*g - d*g)*N*q + lamb*(-1 + N*p + g*(1 - b + (-1 + b + d*(-1 + g) - b*g)*N*p + (d + b*g - d*g)*N*q)))
    return np.array([Q_val_0,Q_val_1])

def Q_multi_prob(transitions, state, predicted_subsidy, discount, threshold=value_iteration_threshold,reward_function='combined',lamb=0.5,
                        match_prob=0.5,match_prob_now=1,num_arms=1):
    """Value iteration when initial and subsequent rewards differ
        We do this through Q Iteration on 3 states
        State 0 and 1 are as normal, while 
        State 2 assumes we get match_prob_now reward, then we transition
        to state 0 or 1, with the same transitions as state 1

    Arguments:
        transitions: Numpy array of size 2x2 (for 2 states x 2 actions)
        state: Integer; which state the arm is currently in
            Will compute Q(s,0) and Q(s,1)
        predicted_subsidy: Float, how much to penalize pulling an arm by
        discount: Float, \gamma, how much to discount future rewards
        threshold: Loop exit condition; when value error <= threshold
            we break
        match_prob: Float; how much reward we get when pulling an arm
            in later time steps
        match_prob_now: Float; how much reward we get when pulling an arm
            in this time step
        num_arms: Total number of arms, N
    
    Returns: 3x2 Numpy array, where Q(2,a) is 
        the rewards for pulling/not pulling an arm now
    """
    assert discount < 1
    assert state == 1

    new_transitions = np.zeros((3,2))
    new_transitions[:2,:2] = transitions 
    new_transitions[2,:] = transitions[1]
    transitions = new_transitions 

    n_states, n_actions = transitions.shape

    value_func = np.array([random.random() for i in range(n_states)])
    difference = np.ones((n_states))
    iters = 0

    def combined_reward(s,a):
        if s == 2:
            s = 1 
            rew = s*a*match_prob_now*(1-lamb) + lamb*s/num_arms - a*predicted_subsidy 
        else:
            rew = s*a*match_prob*(1-lamb) + lamb*s/num_arms - a*predicted_subsidy 
        return rew 

    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        # calculate Q-function
        Q_func = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                r = combined_reward                    
                # transitioning to state = 0
                Q_func[s, a] += (1 - transitions[s, a]) * (r(s, a) + discount * value_func[0])

                # # transitioning to state = 1
                Q_func[s, a] += transitions[s, a] * (r(s, a) + discount * value_func[1])

            value_func[s] = np.max(Q_func[s, :])

        difference = np.abs(orig_value_func - value_func)

    # print(f'q values {Q_func[state, :]}, action {np.argmax(Q_func[state, :])}')
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
    """Generate bounds for upper and lower bounds on penalty
    
    Arguments:
        transitions: 2x2 numpy array; T(s,a,1)
        lamb: Float, lambda value, balancing global, engagement reward
    
    Returns: Tuple of Floats, lower and upper bounds"""

    lb = -1000
    ub = 1000
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
        # TODO: Incorporate this back in 
        # if ub < subsidy_break:
        #     return -10

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

def arm_compute_whittle_multi_prob(transitions, state, discount, subsidy_break, eps=whittle_threshold,reward_function='activity',lamb=0,match_prob=0.5,match_prob_now=1,match_probability_list=[],get_v=False,num_arms=1):
    """
    compute whittle index for a single arm using binary search
    Takes into account the multi-stage rewards (reward now vs. reward later)

    param transitions:
    param eps: epsilon convergence
    returns Whittle index
    """
    lb, ub = get_init_bounds(transitions,lamb) # return lower and upper bounds on WI
    assert state == 1

    while abs(ub - lb) > eps:
        predicted_subsidy = (lb + ub) / 2
        Q_multi = Q_multi_prob(transitions, state, predicted_subsidy, discount,reward_function=reward_function,lamb=lamb,
                    match_prob=match_prob,match_prob_now=match_prob_now,num_arms=num_arms)
        action = np.argmax(Q_multi[2,:])
        if action == 0:
            # optimal action is passive: subsidy is too high
            ub = predicted_subsidy
        elif action == 1:
            # optimal action is active: subsidy is too low
            lb = predicted_subsidy
        else:
            raise Exception(f'action not binary: {action}')
    
    subsidy = (ub + lb) / 2

    return subsidy

def fast_arm_compute_whittle(transitions, state, discount, subsidy_break, eps=whittle_threshold,reward_function='combined',lamb=0.5,match_prob=0.5,num_arms=4):
    """Faster Version of Computing Whittle with multiple probabilities
    Uses explicit formulas for the Whittle index, which were solved through Mathematica
    Doing so avoids expensive Q Iteration
    
    Arguments:  
        transitions: 2x2 numpy array, T(s,a,1)
        state: Integer, 0 or 1
        discount: Float, \gamma, future discount
        subsidy_break: Minimum subsidy, used to compute Whittle when state != 1
    
    Returns: Whittle Index"""

    a = transitions[0,0]
    b = transitions[0,1]
    c = transitions[1,0]
    d = transitions[1,1]
    g = discount 
    e = lamb 
    N = num_arms 
    p = match_prob 
    
    if state == 0:
        answer = ((a - b)*g*(-e + (-1 + e)*N*p))/((1 + b*g - d*g)* N)
    else:
        answer = ((-c + d)*e*g)/((1 + a*g - c*g)*N) + p - e*p
    
    return answer     


def fast_arm_compute_whittle_multi_prob(transitions, state, discount, subsidy_break, eps=whittle_threshold,reward_function='combined',lamb=0.5,match_prob=0.5,match_prob_now=0,num_arms=4):
    """Faster Version of Computing Whittle with multiple probabilities
    Uses explicit formulas for the Whittle index, which were solved through Mathematica
    Doing so avoids expensive Q Iteration
    
    Arguments:  
        transitions: 2x2 numpy array, T(s,a,1)
        state: Integer, 0 or 1
        discount: Float, \gamma, future discount
        subsidy_break: Minimum subsidy, used to compute Whittle when state != 1
    
    Returns: Whittle Index"""
    
    if state != 1:
        return fast_arm_compute_whittle(transitions, state, discount, subsidy_break, eps=whittle_threshold,reward_function='combined',lamb=lamb,match_prob=match_prob,num_arms=num_arms)

    if match_prob_now > match_prob:
        return fast_arm_compute_whittle(transitions, state, discount, subsidy_break, eps=whittle_threshold,reward_function='combined',lamb=lamb,match_prob=match_prob,num_arms=num_arms) + (match_prob_now-match_prob)*(1-lamb)
    else:
        # Use Explicit Formulas from Solving Q Iteration
        a = transitions[0,0]
        c = transitions[1,0]
        d = transitions[1,1]
        N = num_arms 
        top = -((-1 + lamb)*(1 + a*discount)*N*match_prob_now) + c*discount* (-lamb + (-1 + lamb)*N*match_prob)+d*discount* (lamb + lamb* N*(match_prob_now - match_prob) + N*(-match_prob_now + match_prob))
        bottom = N*(1+a*discount-c*discount)
        return top/bottom