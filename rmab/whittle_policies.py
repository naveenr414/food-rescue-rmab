""" Index-Based Algorithms for matching, activity """

import numpy as np

from rmab.utils import Memoizer, contextual_custom_reward
from rmab.compute_whittle import arm_value_iteration_exponential, fast_arm_compute_whittle, fast_arm_compute_whittle_multi_prob, arm_compute_whittle, arm_compute_whittle_multi_prob, fast_compute_whittle_indices
from rmab.utils import binary_to_decimal, custom_reward, one_hot, one_hot_fixed

from copy import deepcopy
import random 
import time
import scipy

def whittle_index(env,state,budget,lamb,memoizer,reward_function="combined",shapley_values=None,contextual=False,match_probs=None,match_prob_now=None,p_matrix=None):
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
        arm_transitions = true_transitions[i//env.volunteers_per_arm]
        if reward_function == "activity":
            check_set_val = memoizer.check_set(arm_transitions+round(p_matrix[i][state[i]],4), state[i])
        else:
            if match_prob_now is not None: 
                check_set_val = memoizer.check_set(arm_transitions+round(p_matrix[i][state[i]],4)+(round(match_prob_now[i],4) + 0.0001)*1000, state[i])
            else:
                check_set_val = memoizer.check_set(arm_transitions+round(p_matrix[i][state[i]],4), state[i])
        if check_set_val != -1:
            state_WI[i] = check_set_val
        else:
            if env.transitions.shape[1] == 2: 
                if match_prob_now is not None:
                    state_WI[i] = fast_arm_compute_whittle_multi_prob(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=match_probability_list[i],match_prob_now=match_prob_now[i],num_arms=len(state))
                else:
                    state_WI[i] = fast_arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=match_probability_list[i],num_arms=len(state))
            else:
                if match_prob_now is not None:
                    state_WI[i] = arm_compute_whittle_multi_prob(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=0,match_prob_now=match_prob_now[i],num_arms=len(state),active_states=env.active_states, p_values=p_matrix[i])
                else:
                    start = time.time()
                    state_WI[i] = arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,p_values=p_matrix[i],num_arms=len(state),active_states=env.active_states)                
                    print("Whittle index took {} time".format(time.time()-start))
            if reward_function == "activity":
                memoizer.add_set(arm_transitions+round(p_matrix[i][state[i]],4), state[i], state_WI[i])
            else:
                if match_prob_now is not None:
                    memoizer.add_set(arm_transitions+round(p_matrix[i][state[i]],4)+(round(match_prob_now[i],4) + 0.0001)*1000, state[i], state_WI[i])
                else:
                    memoizer.add_set(arm_transitions+round(p_matrix[i][state[i]],4), state[i], state_WI[i])
    return state_WI

def shapley_index_custom(env,state,memoizer_shapley = {},idx=-1):
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
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    corresponding_probabilities = match_probabilities
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
        scores.append(custom_reward(state,combo,corresponding_probabilities,env.reward_type,env.reward_parameters))

    scores = np.array(scores)

    num_by_shapley_index = np.zeros(len(state))
    for j,combo in enumerate(combinations):
        action = deepcopy(combo) 

        if idx != -1:
            i = idx 
            if combo[i] == 0:
                action[i] = 1
                shapley_indices[i] += custom_reward(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters) - scores[j]
                num_by_shapley_index[i] += 1
                action[i] = 0
        else:
            for i in range(len(state)):
                if combo[i] == 0:
                    action[i] = 1
                    shapley_indices[i] += custom_reward(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters) - scores[j]
                    num_by_shapley_index[i] += 1
                    action[i] = 0
    shapley_indices /= num_by_shapley_index
    memoizer_shapley[state_str] = shapley_indices

    if idx != -1:
        return shapley_indices[i]
    return shapley_indices, memoizer_shapley

def shapley_index_custom_contexts(env,state,context,memoizer_shapley = {},idx=-1):
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
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    corresponding_probabilities = match_probabilities
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
        scores.append(contextual_custom_reward(state,combo,corresponding_probabilities,env.reward_type,env.reward_parameters,context))

    scores = np.array(scores)

    num_by_shapley_index = np.zeros(len(state))
    for j,combo in enumerate(combinations):
        action = deepcopy(combo) 

        if idx != -1:
            i = idx 
            if combo[i] == 0:
                action[i] = 1
                shapley_indices[i] += contextual_custom_reward(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters,context) - scores[j]
                num_by_shapley_index[i] += 1
                action[i] = 0
        else:
            for i in range(len(state)):
                if combo[i] == 0:
                    action[i] = 1
                    shapley_indices[i] += contextual_custom_reward(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters,context) - scores[j]
                    num_by_shapley_index[i] += 1
                    action[i] = 0
    shapley_indices /= num_by_shapley_index
    memoizer_shapley[state_str] = shapley_indices

    if idx != -1:
        return shapley_indices[i]
    return shapley_indices, memoizer_shapley



def shapley_index_custom_fixed(env,state,memoizer_shapley,arms_pulled,context):
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

    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    corresponding_probabilities = match_probabilities
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

    num_by_shapley_index = np.zeros(len(state))
    start = time.time() 
    scores = []
    for i in range(num_random_combos):
        combo = combinations[i]
        scores.append(contextual_custom_reward(state,combo,corresponding_probabilities,env.reward_type,env.reward_parameters,context))

    scores = np.array(scores)

    for j,combo in enumerate(combinations):
        action = deepcopy(combo) 
        for i in range(len(state)):
            if combo[i] == 0:
                action[i] = 1
                shapley_indices[i] += contextual_custom_reward(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters,context) - scores[j]
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
    n_states = env.transitions.shape[1]

    if memory is None:
        reward_matrix = np.zeros((N,n_states,2))

        for i in range(N):
            for j in range(n_states):
                if j in env.active_states:
                    reward_matrix[i,j] += lamb/N

        whittle_matrix = np.zeros((N,n_states))
        for i in range(N):
            for j in range(n_states):
                whittle_matrix[i] = fast_compute_whittle_indices(env.transitions[i//env.volunteers_per_arm],reward_matrix[i],env.discount)
    else:
        whittle_matrix = memory 
    
    state_WI = [whittle_matrix[i][state[i]] for i in range(N)]

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (whittle_matrix)  

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
    n_states = env.transitions.shape[1]

    if memory is None:
        p_matrix = np.zeros((N,n_states))

        for i in range(N):
            for s in range(n_states):
                default_state = [env.worst_state for _ in range(N)]
                default_state[i] = s
                p_matrix[i,s] = custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters)

        reward_matrix = np.zeros((N,n_states,2))

        for i in range(N):
            for j in range(n_states):
                if j in env.active_states:
                    reward_matrix[i,j] += lamb/N
                reward_matrix[i,j,1] += (1-lamb)*p_matrix[i,j]

        whittle_matrix = np.zeros((N,n_states))
        for i in range(N):
            for j in range(n_states):
                whittle_matrix[i] = fast_compute_whittle_indices(env.transitions[i//env.volunteers_per_arm],reward_matrix[i],env.discount)
    else:
        whittle_matrix = memory 
    
    state_WI = [whittle_matrix[i][state[i]] for i in range(N)]

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (whittle_matrix)  

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
    n_states = env.transitions.shape[1]

    N = len(state) 
    n_states = env.transitions.shape[1]

    if memory is None:
        p_matrix = np.zeros((N,n_states))

        for i in range(N):
            for s in range(n_states):
                default_state = [env.worst_state for _ in range(N)]
                default_state[i] = s
                p_matrix[i,s] = custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters)

        reward_matrix = np.zeros((N,n_states,2))

        for i in range(N):
            for j in range(n_states):
                if j in env.active_states:
                    reward_matrix[i,j] += lamb/N
                reward_matrix[i,j,1] += (1-lamb)*p_matrix[i,j]
    else:
        reward_matrix = memory 

    action = np.zeros(N, dtype=np.int8)
    pulled_arms = []

    for _ in range(budget):
        if len(pulled_arms) > 0:
            pulled_action = one_hot_fixed(pulled_arms[0],len(state),pulled_arms)
            default_custom_reward = contextual_custom_reward(state,pulled_action,np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.context)
        else:
            pulled_action = [0 for i in range(len(state))]
            default_custom_reward = 0
        match_prob_now = []
        state_WI = []

        for i in range(N):
            if i not in pulled_arms:
                new_action = deepcopy(pulled_action)
                new_action[i] = 1

                # TODO: Make t his an option so we can see the impact 
                match_prob_now= contextual_custom_reward(state,new_action,np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.context)-default_custom_reward

                new_transitions = np.zeros((n_states+1,2,n_states+1))
                new_transitions[:n_states,:,:n_states] = env.transitions[i//env.volunteers_per_arm]
                new_transitions[n_states] = new_transitions[state[i]]

                new_reward_matrix = np.zeros((n_states+1,2))
                new_reward_matrix[:n_states] = reward_matrix[i]

                if i in env.active_states:
                    new_reward_matrix[n_states] += lamb/N 
                new_reward_matrix[n_states,1] += (1-lamb)*match_prob_now 
                whittle_compute = fast_compute_whittle_indices(new_transitions,new_reward_matrix,env.discount)[-1]
                state_WI.append(whittle_compute)
                
            else:
                state_WI.append(-100) 
        state_WI = np.array(state_WI)
        argmax_val = np.argmax(state_WI)
        action[argmax_val] = 1 

        pulled_arms.append(argmax_val)

    return action, (reward_matrix) 

def contextual_whittle_policy(env,state,budget,lamb,memory,per_epoch_results):
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
    n_states = env.transitions.shape[1]

    if memory is None:    
        # Construct transitions of size |S|*|D|, where we get |D| samples
        num_samples = 10
        random_contexts = np.array([[random.random() for i in range(len(state))] for _ in range(num_samples)])
        new_reward_matrix = np.zeros((N,num_samples*n_states+1,2))
        transitions = np.zeros((N,num_samples*n_states+1,2,num_samples*n_states+1))

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    if j in env.active_states:
                        new_reward_matrix[i,idx] += lamb/N
                    default_state = [env.worst_state for _ in range(N)]
                    default_state[i] = j
                    new_reward_matrix[i,idx,1] += (1-lamb)*contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,random_contexts[k])
                    transitions[i][idx][:,:-1] = np.repeat(env.transitions[i][j], num_samples,axis=1)
                    transitions[i][idx] /= num_samples 

            if state[i] in env.active_states:
                new_reward_matrix[i,-1] += lamb/N
    
            transitions[i][-1] = transitions[i][state[i]*num_samples]

        baseline_whittles = np.zeros((N,n_states))
        for i in range(N):
            for s in range(n_states):
                transitions[i][-1] = transitions[i][s*num_samples]
                baseline_whittles[i][s] = fast_compute_whittle_indices(transitions[i],new_reward_matrix[i],env.discount)[-1]

        scale_whittles = np.zeros((N,n_states))
        for i in range(N):
            for s in range(n_states):
                transitions[i][-1] = transitions[i][s*num_samples]
                new_reward_matrix[i,-1,1] += 1
                scale_whittles[i][s] = fast_compute_whittle_indices(transitions[i],new_reward_matrix[i],env.discount)[-1]
                new_reward_matrix[i,-1,1] -= 1
    else:
        baseline_whittles,scale_whittles = memory 
    
    rewards = np.zeros(N)

    for i in range(N):
        default_state = [env.worst_state for _ in range(N)]
        default_state[i] = state[i]
        rewards[i] = (1-lamb)*contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.context)
    
    state_WI = [baseline_whittles[i][state[i]]+scale_whittles[i][state[i]]*rewards[i] for i in range(N)]

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (baseline_whittles,scale_whittles)  

def contextual_shapley_policy(env,state,budget,lamb,memory,per_epoch_results):
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
    n_states = env.transitions.shape[1]

    if memory is None:    
        # Construct transitions of size |S|*|D|, where we get |D| samples
        num_samples = 10
        random_contexts = np.array([[random.random() for i in range(len(state))] for _ in range(num_samples)])
        new_reward_matrix = np.zeros((N,num_samples*n_states+1,2))
        transitions = np.zeros((N,num_samples*n_states+1,2,num_samples*n_states+1))

        context_shapley_values = np.zeros((N,num_samples*n_states))

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    default_state = [env.best_state for _ in range(N)]
                    default_state[i] = j
                    context = random_contexts[k]
                    res = shapley_index_custom_contexts(env,default_state,context,idx=i)
                    context_shapley_values[i,idx] = res

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    if j in env.active_states:
                        new_reward_matrix[i,idx] += lamb/N
                    default_state = [env.worst_state for _ in range(N)]
                    default_state[i] = j
                    new_reward_matrix[i,idx,1] += (1-lamb)*context_shapley_values[i,idx]
                    transitions[i][idx][:,:-1] = np.repeat(env.transitions[i][j], num_samples,axis=1)
                    transitions[i][idx] /= num_samples 

            if state[i] in env.active_states:
                new_reward_matrix[i,-1] += lamb/N
    
            transitions[i][-1] = transitions[i][state[i]*num_samples]
        baseline_whittles = np.zeros((N,n_states))
        for i in range(N):
            for s in range(n_states):
                transitions[i][-1] = transitions[i][s*num_samples]
                baseline_whittles[i][s] = fast_compute_whittle_indices(transitions[i],new_reward_matrix[i],env.discount)[-1]

        scale_whittles = np.zeros((N,n_states))
        for i in range(N):
            for s in range(n_states):
                transitions[i][-1] = transitions[i][s*num_samples]
                new_reward_matrix[i,-1,1] += 1
                scale_whittles[i][s] = fast_compute_whittle_indices(transitions[i],new_reward_matrix[i],env.discount)[-1]
                new_reward_matrix[i,-1,1] -= 1
    else:
        baseline_whittles,scale_whittles = memory 
    
    rewards = np.zeros(N)

    for i in range(N):
        default_state = [env.best_state for _ in range(N)]
        default_state[i] = state[i]
        rewards[i] = (1-lamb)*shapley_index_custom_contexts(env,default_state,env.context,idx=i)
    
    state_WI = [baseline_whittles[i][state[i]]+scale_whittles[i][state[i]]*rewards[i] for i in range(N)]

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (baseline_whittles,scale_whittles)  



 
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
    n_states = env.transitions.shape[1]

    if memory is None:
        u_matrix = np.zeros((N,n_states))

        for s in range(n_states):
            for i in range(N):
                curr_state = [env.best_state for _ in range(N)]
                curr_state[i] = s
                u_matrix[i,s] = shapley_index_custom(env,curr_state,{},idx=i)
        reward_matrix = np.zeros((N,n_states,2))

        for i in range(N):
            for j in range(n_states):
                if j in env.active_states:
                    reward_matrix[i,j] += lamb/N
                reward_matrix[i,j,1] += (1-lamb)*u_matrix[i,j]
    else:
        reward_matrix = memory 


    action = np.zeros(N, dtype=np.int8)
    pulled_arms = []

    for _ in range(budget):
        if len(pulled_arms) > 0:
            pulled_action = one_hot_fixed(pulled_arms[0],len(state),pulled_arms)
        else:
            pulled_action = [0 for i in range(len(state))]
        match_prob_now = []
        state_WI = []
        match_prob_all = np.array(shapley_index_custom_fixed(env,state,{},pulled_arms,env.context)[0])

        for i in range(N):
            if i not in pulled_arms:
                new_action = deepcopy(pulled_action)
                new_action[i] 
                match_prob_now = match_prob_all[i]

                new_transitions = np.zeros((n_states+1,2,n_states+1))
                new_transitions[:n_states,:,:n_states] = env.transitions[i//env.volunteers_per_arm]
                new_transitions[n_states] = new_transitions[state[i]]

                new_reward_matrix = np.zeros((n_states+1,2))
                new_reward_matrix[:n_states] = reward_matrix[i]

                if i in env.active_states:
                    new_reward_matrix[n_states] += lamb/N 
                new_reward_matrix[n_states,1] += (1-lamb)*match_prob_now 
                state_WI.append(fast_compute_whittle_indices(new_transitions,new_reward_matrix,env.discount)[-1])
            else:
                state_WI.append(-100) 
        state_WI = np.array(state_WI)
        argmax_val = np.argmax(state_WI)
        action[argmax_val] = 1 

        pulled_arms.append(argmax_val)

    return action, (reward_matrix) 


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
    n_states = env.transitions.shape[1]

    if memory is None:
        u_matrix = np.zeros((N,n_states))

        for s in range(n_states):
            for i in range(N):
                curr_state = [env.best_state for _ in range(N)]
                curr_state[i] = s
                u_matrix[i,s] = shapley_index_custom(env,curr_state,{},idx=i)
        reward_matrix = np.zeros((N,n_states,2))

        for i in range(N):
            for j in range(n_states):
                if j in env.active_states:
                    reward_matrix[i,j] += lamb/N
                reward_matrix[i,j,1] += (1-lamb)*u_matrix[i,j]

        whittle_matrix = np.zeros((N,n_states))
        for i in range(N):
            for j in range(n_states):
                whittle_matrix[i] = fast_compute_whittle_indices(env.transitions[i//env.volunteers_per_arm],reward_matrix[i],env.discount)
    else:
        whittle_matrix = memory 
    
    state_WI = [whittle_matrix[i][state[i]] for i in range(N)]

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (whittle_matrix)  
