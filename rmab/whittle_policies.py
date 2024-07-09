""" Index-Based Algorithms for matching, activity """

import numpy as np

from rmab.utils import Memoizer
from rmab.compute_whittle import arm_value_iteration_exponential, fast_arm_compute_whittle, fast_arm_compute_whittle_multi_prob
from rmab.utils import binary_to_decimal, custom_reward, one_hot, one_hot_fixed, custom_reward_contextual, get_average_context, avg_reward_contextual

from copy import deepcopy
import random 
import time
import scipy

def whittle_index(env,state,budget,lamb,memoizer,reward_function="combined",shapley_values=None,contextual=False,match_probs=None,match_prob_now=None):
    """Get the Whittle indices for each agent in a given state
    
    Arguments:
        env: Simualtor RMAB environment
        state: Numpy array of 0-1 for each volunteer, s_{i}
        budget: Max arms to pick, K
        lamb: Float, balancing matching and activity, \alpha 
        memoizer: Object that stores previous Whittle index computations
    
    Returns: List of Whittle indices for each arm, w_{i}(s_{i})"""
    N = len(state) 

    if reward_function == "activity":
        match_probability_list = [0 for i in range(len(env.agent_idx))]
    elif shapley_values != None:
        match_probability_list = np.array(shapley_values)
    elif contextual or match_probs is not None:
        match_probability_list = match_probs
    else:
        match_probability_list = np.array(env.match_probability_list)[env.agent_idx]

    true_transitions = env.transitions 
    discount = env.discount 

    state_WI = np.zeros((N))
    min_chosen_subsidy = -1 
    for i in range(N):
        arm_transitions = true_transitions[i//env.volunteers_per_arm, :, :, 1]
        if reward_function == "activity":
            check_set_val = memoizer.check_set(arm_transitions, state[i])
        else:
            if match_prob_now is not None: 
                check_set_val = memoizer.check_set(arm_transitions+round(match_probability_list[i],4)+(round(match_prob_now[i],4) + 0.0001)*1000, state[i])
            else:
                check_set_val = memoizer.check_set(arm_transitions+round(match_probability_list[i],4), state[i])
        if check_set_val != -1:
            state_WI[i] = check_set_val
        else:
            if match_prob_now is not None:
                state_WI[i] = fast_arm_compute_whittle_multi_prob(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=match_probability_list[i],match_prob_now=match_prob_now[i],num_arms=len(state))
            else:
                state_WI[i] = fast_arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=match_probability_list[i],num_arms=len(state))
            if reward_function == "activity":
                memoizer.add_set(arm_transitions, state[i], state_WI[i])
            else:
                if match_prob_now is not None:
                    memoizer.add_set(arm_transitions+round(match_probability_list[i],4)+(round(match_prob_now[i],4) + 0.0001)*1000, state[i], state_WI[i])
                else:
                    memoizer.add_set(arm_transitions+round(match_probability_list[i],4), state[i], state_WI[i])
    return state_WI

def shapley_index_custom(env,state,memoizer_shapley = {}):
    """Compute the Shapley index for matching; how much
        does match probability increase when using some particular arm
        u_{i}(s_{i})
        
    Arguments:
        env: RMAB Simulator environment
        state: Numpy array of 0-1 states for each volunteer, s_{i}
        memoizer_shapley: Dictionary, to store previously computed Shapley indices
        
    Returns: Two things, shapley index, and updated dictionary"""

    shapley_indices = [0 for i in range(len(state))]
    state_str = "".join([str(i) for i in state])
    if state_str in memoizer_shapley:
        return memoizer_shapley[state_str], memoizer_shapley
    state_1 = [i for i in range(len(state)) if state[i] != 0]
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    corresponding_probabilities = match_probabilities[state_1]
    num_random_combos = env.shapley_iterations 

    combinations = np.zeros((num_random_combos, len(corresponding_probabilities)), dtype=int)
    budget = env.budget 
    if len(corresponding_probabilities) <= env.budget-1:
        if len(corresponding_probabilities) == 1:
            return match_probabilities * state, memoizer_shapley
        else: 
            budget = 2
    budget_probs = np.array([scipy.special.comb(len(corresponding_probabilities),k) for k in range(0,budget)])
    budget_probs /= np.sum(budget_probs)

    for i in range(num_random_combos):
        k = random.choices(list(range(len(budget_probs))), weights=budget_probs,k=1)[0]
        ones_indices = random.sample(list(range(len(corresponding_probabilities))),k)
        combinations[i, ones_indices] = 1

    state = [int(i) for i in state]
    scores = []
    for i in range(num_random_combos):
        combo = combinations[i]

        if env.use_context:
            scores.append(avg_reward_contextual(state,combo,corresponding_probabilities,env.reward_type,env.reward_parameters))
        else:
            scores.append(custom_reward(state,combo,corresponding_probabilities,env.reward_type,env.reward_parameters))
    scores = np.array(scores)

    num_by_shapley_index = np.zeros(len(state_1))
    for j,combo in enumerate(combinations):
        action = deepcopy(combo) 
        for i in range(len(state_1)):
            if combo[i] == 0:
                action[i] = 1

                if env.use_context:
                    shapley_indices[i] += avg_reward_contextual(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters) - scores[j]
                else:
                    shapley_indices[i] += custom_reward(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters) - scores[j]
                num_by_shapley_index[i] += 1
                action[i] = 0
    shapley_indices /= num_by_shapley_index
    memoizer_shapley[state_str] = shapley_indices

    return shapley_indices, memoizer_shapley

def compute_regression(env):
    """Compute the Shapley index for matching; how much
        does match probability increase when using some particular arm
        u_{i}(s_{i})
        
    Arguments:
        env: RMAB Simulator environment
        state: Numpy array of 0-1 states for each volunteer, s_{i}
        memoizer_shapley: Dictionary, to store previously computed Shapley indices
        
    Returns: Two things, shapley index, and updated dictionary"""

    N = env.cohort_size*env.volunteers_per_arm
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    num_random_combos = env.shapley_iterations 

    combinations = np.zeros((num_random_combos, N), dtype=int)
    budget = env.budget 

    budget_probs = np.array([scipy.special.comb(N,k) for k in range(0,budget+1)])
    budget_probs /= np.sum(budget_probs)

    for i in range(num_random_combos):
        k = random.choices(list(range(len(budget_probs))), weights=budget_probs,k=1)[0]
        ones_indices = random.sample(list(range(N)),k)
        combinations[i, ones_indices] = 1

    unique_X, indices = np.unique(combinations, axis=0, return_index=True)

    # Sort indices to maintain original order
    sorted_indices = np.sort(indices)

    state = [1 for i in range(N)]
    scores = []
    for i in range(num_random_combos):
        combo = combinations[i]
        scores.append(avg_reward_contextual(state,combo,match_probabilities,env.reward_type,env.reward_parameters))
    scores = np.array(scores)

    X = combinations 
    Y = scores

    X = X[indices]
    Y = Y[indices]

    # Calculate the least squares solution
    # Theta_best = (X_b.T * X_b)^-1 * X_b.T * Y
    theta_best = np.linalg.inv(X.T @ X) @ X.T @ Y

    return theta_best

def shapley_index_custom_fixed(env,state,memoizer_shapley,arms_pulled):
    """Compute the Shapley index for matching; how much
        does match probability increase when using some particular arm
        Assume that some arms were already pulled 
        
    Arguments:
        env: RMAB Simulator environment
        state: Numpy array of 0-1 states for each volunteer
        memoizer_shapley: Dictionary, to store previously computed Shapley indices
        arms_pulled: Which arms were already pulled 
        
    Returns: Two things, shapley index, and updated dictionary"""

    shapley_indices = [0 for i in range(len(state))]
    state_str = "".join([str(i) for i in state])

    if state_str in memoizer_shapley:
        return memoizer_shapley[state_str], memoizer_shapley

    state_1 = [i for i in range(len(state)) if state[i] != 0]
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    corresponding_probabilities = match_probabilities[state_1]
    num_random_combos = env.shapley_iterations

    combinations = np.zeros((num_random_combos, len(corresponding_probabilities)), dtype=int)

    budget = env.budget-len(arms_pulled)

    # Fix for when the number of combinations is small (with respect to the budget)
    # In that scenario, we can essentially just manually compute
    if len(corresponding_probabilities) <= env.budget-1:
        if len(corresponding_probabilities) == 1:
            return match_probabilities * state, memoizer_shapley
        else: 
            budget = 2

    budget_probs = np.array([scipy.special.comb(len(corresponding_probabilities),k) for k in range(0,budget)])
    budget_probs /= np.sum(budget_probs)

    set_arms_pulled = set(arms_pulled)
    arms_not_pulled = [i for i in range(len(state)) if i not in set_arms_pulled]

    for i in range(num_random_combos):
        k = random.choices(list(range(len(budget_probs))), weights=budget_probs,k=1)[0]
        ones_indices = random.sample(arms_not_pulled,k)
        combinations[i, ones_indices] = 1
        combinations[i,arms_pulled] = 1

    state = [int(i) for i in state]

    num_by_shapley_index = np.zeros(len(state_1))
    start = time.time() 
    scores = []
    for i in range(num_random_combos):
        combo = combinations[i]
        scores.append(custom_reward_contextual(state,combo,corresponding_probabilities,env.reward_type,env.reward_parameters,env.context))

    scores = np.array(scores)

    for j,combo in enumerate(combinations):
        action = deepcopy(combo) 
        for i in range(len(state_1)):
            if combo[i] == 0:
                action[i] = 1
                shapley_indices[i] += custom_reward_contextual(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters,env.context) - scores[j]
                num_by_shapley_index[i] += 1
                action[i] = 0
        if time.time()-start > env.time_limit:
            break 
    shapley_indices = shapley_indices / num_by_shapley_index
    shapley_indices = np.nan_to_num(shapley_indices,0)
    shapley_indices[shapley_indices == np.inf] = 0

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

        for i in range(2):
            s = [i for _ in range(len(state))]
            whittle_index(env,s,budget,lamb,memoizer,reward_function="activity")
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

        if env.use_context:
            match_probs = [avg_reward_contextual(one_hot(i,len(state)),one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters) for i in range(len(state))]
        else:
            match_probs = [custom_reward(one_hot(i,len(state)),one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters) for i in range(len(state))]
        print("Match probs {}".format(match_probs))
        for i in range(2):
            s = [i for _ in range(len(state))]
            whittle_index(env,s,budget,lamb,memoizer,reward_function="combined",match_probs=match_probs)
    else:
        memoizer, match_probs = memory 
        
    state_WI = whittle_index(env,state,budget,lamb,memoizer,match_probs=match_probs)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (memoizer,match_probs)  

def whittle_iterative_policy(env,state,budget,lamb,memory,per_epoch_results):
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
        match_probs = [custom_reward(one_hot(i,len(state)),one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters) for i in range(len(state))]
        for i in range(2):
            s = [i for _ in range(len(state))]
            whittle_index(env,s,budget,lamb,memoizer,reward_function="combined",match_probs=match_probs)
    
    else:
        memoizer, match_probs = memory 

    action = np.zeros(N, dtype=np.int8)
    pulled_arms = []

    start = time.time() 

    for _ in range(budget):
        if len(pulled_arms) > 0:
            arms_0_1 = one_hot_fixed(pulled_arms[0],len(state),pulled_arms)
            default_custom_reward = custom_reward_contextual(arms_0_1,arms_0_1,np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.context)
        else:
            default_custom_reward = 0
        match_prob_now = [custom_reward_contextual(one_hot_fixed(i,len(state),pulled_arms),one_hot_fixed(i,len(state),pulled_arms),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.context)-default_custom_reward for i in range(len(state))]
        match_prob_now = np.array(match_prob_now)

        state_WI = whittle_index(env,state,budget,lamb,memoizer,match_probs=match_probs,match_prob_now=match_prob_now)
        
        state_WI[action == 1] = -100
        argmax_val = np.argmax(state_WI)
        action[argmax_val] = 1 

        pulled_arms.append(argmax_val)

        if time.time()-start > env.time_limit:
            break 

    return action, (memoizer,match_probs) 
 
def greedy_whittle_iterative_policy(env,state,budget,lamb,memory,per_epoch_results):
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

    action = np.zeros(N, dtype=np.int8)
    pulled_arms = []

    for _ in range(budget):
        match_probability_list = np.array([custom_reward(one_hot_fixed(i,len(state),pulled_arms),one_hot_fixed(i,len(state),pulled_arms),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters) for i in range(len(state))])
        state_WI = whittle_index(env,state,budget,lamb,memoizer,match_probs=match_probability_list,reward_function="activity")
        state_WI *= lamb 
        state_WI += (1-lamb)*match_probability_list
        state_WI[action == 1] = -100

        argmax_val = np.argmax(state_WI)
        action[argmax_val] = 1 
        pulled_arms.append(argmax_val)

    return action, memoizer 

def shapley_whittle_iterative_policy(env,state,budget,lamb,memory,per_epoch_results):
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
        memory_whittle = Memoizer('optimal')
        match_probs = np.array(shapley_index_custom(env,np.ones(len(state)),{})[0])
        for i in range(2):
            s = [i for _ in range(len(state))]
            whittle_index(env,s,budget,lamb,memory_whittle,reward_function="combined",match_probs=match_probs)

    else:
        memory_whittle, match_probs = memory 

    action = np.zeros(N, dtype=np.int8)
    pulled_arms = []

    start = time.time() 

    for _ in range(budget):
        match_prob_now = np.array(shapley_index_custom_fixed(env,np.ones(len(state)),{},pulled_arms)[0])

        state_WI = whittle_index(env,state,budget,lamb,memory_whittle,reward_function="combined",match_probs=match_probs,match_prob_now=match_prob_now)
        state_WI[action == 1] = -100

        argmax_val = np.argmax(state_WI)
        if state_WI[argmax_val] <= 0:
            break 

        action[argmax_val] = 1 
        pulled_arms.append(argmax_val)

        if time.time()-start > env.time_limit:
            break 
    return action, (memory_whittle,match_probs) 


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
        z = 1/0

    max_action = np.argmax(Q_vals[state_rep])
    binary_val = bin(max_action)[2:].zfill(N)

    action = np.zeros(N, dtype=np.int8)
    action = np.array([int(i) for i in binary_val])

    rew = custom_reward(state,action,np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters)

    return action, None


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

    match_probabilities = np.array([custom_reward(one_hot(i,len(state)),one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters) for i in range(len(state))])

    state_WI += (1-lamb)*match_probabilities

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, memoizer 

def shapley_whittle_custom_policy(env,state,budget,lamb,memory, per_epoch_results):
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
        memory_shapley = np.array(shapley_index_custom(env,np.ones(len(state)),{})[0])
    
        for i in range(2):
            s = [i for _ in range(len(state))]
            whittle_index(env,s,budget,lamb,memory_whittle,reward_function="combined",match_probs=memory_shapley)
    else:
        memory_whittle, memory_shapley = memory 

    state_WI = whittle_index(env,state,budget,lamb,memory_whittle,reward_function="combined",match_probs=memory_shapley)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (memory_whittle, memory_shapley)

def regression_policy(env,state,budget,lamb,memory, per_epoch_results):
    """Combination of the Whittle index + Regressed values
    
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
        memory_shapley = compute_regression(env)
    
        for i in range(2):
            s = [i for _ in range(len(state))]
            whittle_index(env,s,budget,lamb,memory_whittle,reward_function="combined",match_probs=memory_shapley)
    else:
        memory_whittle, memory_shapley = memory 

    state_WI = whittle_index(env,state,budget,lamb,memory_whittle,reward_function="combined",match_probs=memory_shapley)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (memory_whittle, memory_shapley)

