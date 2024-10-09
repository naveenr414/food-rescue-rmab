""" Index-Based Algorithms for matching, activity """

import numpy as np

from rmab.utils import contextual_custom_reward, custom_reward, one_hot, one_hot_fixed, shapley_index_custom_contexts, shapley_index_custom_fixed, compute_u_matrix
from rmab.compute_whittle import fast_compute_whittle_indices
from rmab.baseline_policies import compute_reward_matrix, compute_p_matrix

from copy import deepcopy, copy
import time 

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
        reward_matrix = compute_reward_matrix(env,N,1)
        whittle_matrix = np.zeros((N,n_states))
        for i in range(N):
            whittle_matrix[i] = fast_compute_whittle_indices(env.transitions[i//env.volunteers_per_arm],reward_matrix[i],env.discount)
    else:
        whittle_matrix = memory 
    
    state_WI = [whittle_matrix[i][state[i]] for i in range(N)]
    sorted_WI = np.argsort(state_WI)[::-1]

    filtered_WI = [i for i in sorted_WI if state_WI[i] >= 0]

    action = np.zeros(N, dtype=np.int8)
    action[filtered_WI[:budget]] = 1
    return action, whittle_matrix


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
        p_matrix = compute_p_matrix(env,N)
        reward_matrix = compute_reward_matrix(env,N,lamb)
        reward_matrix[:,:,1] += (1-lamb)*p_matrix
        whittle_matrix = np.zeros((N,n_states))
        for i in range(N):
            whittle_matrix[i] = fast_compute_whittle_indices(env.transitions[i//env.volunteers_per_arm],reward_matrix[i],env.discount)
    else:
        whittle_matrix = memory 
    
    
    state_WI = [whittle_matrix[i][state[i]] for i in range(N)]
    sorted_WI = np.argsort(state_WI)[::-1]
    filtered_WI = [i for i in sorted_WI if state_WI[i] >= 0]

    action = np.zeros(N, dtype=np.int8)
    action[filtered_WI[:budget]] = 1

    return action, whittle_matrix


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
        u_matrix =  compute_u_matrix(env,N,n_states)
        reward_matrix = compute_reward_matrix(env,N,lamb)
        reward_matrix[:,:,1] += (1-lamb)*u_matrix
        whittle_matrix = np.zeros((N,n_states))

        for i in range(N):
            for j in range(n_states):
                if j in env.active_states:
                    reward_matrix[i,j] += lamb/N
                reward_matrix[i,j,1] += (1-lamb)*u_matrix[i,j]

        whittle_matrix = np.zeros((N,n_states))
        for i in range(N):
            whittle_matrix[i] = fast_compute_whittle_indices(env.transitions[i//env.volunteers_per_arm],reward_matrix[i],env.discount)    
    else:
        whittle_matrix = memory 
    
    state_WI = np.array([whittle_matrix[i][state[i]] for i in range(N)])


    sorted_WI = np.argsort(state_WI)[::-1]

    filtered_WI = [i for i in sorted_WI if state_WI[i] >= 0]

    action = np.zeros(N, dtype=np.int8)
    action[filtered_WI[:budget]] = 1

    return action, whittle_matrix

def get_whittle_function(contextual):
    """Returns a function that computes the Linear marginal rewards per arm, given the pulled arms
    
    Arguments:
        contextual: Boolean
    
    Returns: A function, which, given the pulled arms, computes marginal rewards for each arm"""

    def run_function(env,state,N,pulled_arms,shapley_memoizer):
        return whittle_match_prob_now(env,state,N,pulled_arms,contextual=contextual)

    return run_function

def whittle_match_prob_now(env,state,N,pulled_arms,contextual=True):
    """Compute the Marginal reward for each arm given a set of pulled arms
    
    Arguments:
        env: RMAB simulator environment
        state: Numpy array of 0-1 states for each arm
        N: Number of ARms
        pulled_arms: List of pulled arms, by indices
        contextual: Boolean, whether to use the contextual variant of the Shapley policy"""

    if len(pulled_arms) > 0:
        pulled_action = one_hot_fixed(pulled_arms[0],len(state),pulled_arms)

        if contextual:
            default_custom_reward = contextual_custom_reward(state,pulled_action,np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.active_states,env.context)
        else:
            default_custom_reward = custom_reward(state,pulled_action,np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.active_states)

    else:
        pulled_action = [0 for i in range(len(state))]
        default_custom_reward = 0

    match_prob_all = []
    for i in range(N):
        new_action = copy(pulled_action)
        new_action[i] = 1

        if contextual:
            match_prob_all.append(contextual_custom_reward(state,new_action,np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.active_states,env.context)-default_custom_reward)
        else:
            match_prob_all.append(custom_reward(state,new_action,np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.active_states)-default_custom_reward)
    return match_prob_all

def get_shapley_function(contextual):
    """Returns a function that computes the Shapley values per arm, given the pulled arms
    
    Arguments:
        contextual: Boolean
    
    Returns: A function, which, given the pulled arms, computes Shapley values for each arm"""
    def run_function(env,state,N,pulled_arms,shapley_memoizer={}):
        return shapley_match_prob_now(env,state,N,pulled_arms,shapley_memoizer,contextual=contextual)
    return run_function

def shapley_match_prob_now(env,state,N,pulled_arms,shapley_memoizer,contextual=True):
    """Compute the Shapley values for each arm given a set of pulled arms
    
    Arguments:
        env: RMAB simulator environment
        state: Numpy array of 0-1 states for each arm
        N: Number of ARms
        pulled_arms: List of pulled arms, by indices
        shapley_memoizer: Dictionary, which stores precomputed Shapley values
        contextual: Boolean, whether to use the contextual variant of the Shapley policy"""
    
    if contextual:
        match_prob_all = shapley_index_custom_fixed(env,state,shapley_memoizer,pulled_arms,env.context)[0]
    else:
        match_prob_all = shapley_index_custom_fixed(env,state,shapley_memoizer,pulled_arms,np.array(env.match_probability_list)[env.agent_idx])[0]
    return match_prob_all

def iterative_policy_skeleton(env,state,budget,lamb,memory,per_epoch_results,match_prob_now_function,reward_matrix):
    """Skeleton for running any iterative policy; which can be used for 
    Shapley or Linear-Whittle Iterative Policies
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Information on previously computed Whittle indices, the memoizer
        match_prob_now_function: Soem funcion that returns either the match probability
            given some arms pulled, or the additional Shapley value
        reward_matrix: Either the p_matrix or the u_matrix, to compute Linear- and Shapley-Whittle
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""
    
    N = len(state)
    n_states = env.transitions.shape[1]
    
    action = np.zeros(N, dtype=np.int8)
    pulled_arms = []

    start = time.time() 
    tot = 0
    tot_2 = 0
    tot_3 = 0
    computed_values = memory[1]
    shapley_memoizer = memory[2]

    for _ in range(budget):
        temp = time.time()
        match_prob_now_list = match_prob_now_function(env,state,N,pulled_arms,shapley_memoizer)
        tot_3 += time.time()-temp
        state_WI = []

        for i in range(N):
            if i not in pulled_arms:
                temp = time.time()
                match_prob_now = match_prob_now_list[i]
                new_transitions = np.zeros((n_states+1,2,n_states+1))
                new_transitions[:n_states,:,:n_states] = env.transitions[i//env.volunteers_per_arm]
                new_transitions[n_states] = new_transitions[state[i]]

                new_reward_matrix = np.zeros((n_states+1,2))
                new_reward_matrix[:n_states] = reward_matrix[i]

                if i in env.active_states:
                    new_reward_matrix[n_states] += lamb/N 
                new_reward_matrix[n_states,1] += (1-lamb)*match_prob_now 
                tot += (time.time()-temp)
                temp = time.time() 

                whittle_compute = fast_compute_whittle_indices(new_transitions,new_reward_matrix,env.discount,computed_values=computed_values)[-1]
                state_WI.append(whittle_compute)
                tot_2 += (time.time()-temp)
            else:
                state_WI.append(-100) 
        state_WI = np.array(state_WI)
        argmax_val = np.argmax(state_WI)
        action[argmax_val] = 1 

        pulled_arms.append(argmax_val)

    return action, (reward_matrix,computed_values,shapley_memoizer) 


def whittle_iterative_policy(env,state,budget,lamb,memory,per_epoch_results,contextual=True):
    """Whittle index policy which iteratively selects arms
    Essentially by first computing the Whittle indices, then one by one updating
    As arms are selected
    
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
        p_matrix = compute_p_matrix(env,N)
        reward_matrix = compute_reward_matrix(env,N,lamb)
        reward_matrix[:,:,1] += (1-lamb)*p_matrix
        computed_values = {}
        shapley_memoizer = {}
        memory = reward_matrix, computed_values, shapley_memoizer
    else:
        reward_matrix, computed_values, shapley_memoizer = memory 

    whittle_function = get_whittle_function(contextual)

    return iterative_policy_skeleton(env,state,budget,lamb,memory,per_epoch_results,whittle_function,reward_matrix)

def non_contextual_whittle_iterative_policy(env,state,budget,lamb,memory,per_epoch_results):
    """Variant of the Iterative Whittle policy, where the current context is unknown"""
    return whittle_iterative_policy(env,state,budget,lamb,memory,per_epoch_results,contextual=False)

def shapley_whittle_iterative_policy(env,state,budget,lamb,memory,per_epoch_results,contextual=True):
    """Whittle index policy based on computing the subsidy for each arm
    This approximates the problem as the sum of Shapley values for each arm
    Does so iteratively, by first selecting highest Whittle index, then updating
    Whittle indices
    
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
        u_matrix = compute_u_matrix(env,N,n_states)
        reward_matrix = compute_reward_matrix(env,N,lamb)
        reward_matrix[:,:,1] += (1-lamb)*u_matrix
        computed_values = {}
        shapley_memoizer = {}
        memory = reward_matrix, computed_values, shapley_memoizer
    else:
        reward_matrix, computed_values, shapley_memoizer = memory 

    shapley_function = get_shapley_function(contextual)

    return iterative_policy_skeleton(env,state,budget,lamb,memory,per_epoch_results,shapley_function,reward_matrix)

def non_contextual_shapley_whittle_iterative_policy(env,state,budget,lamb,memory,per_epoch_results):
    """Variant of the Iterative Policy, where the current context is unknown"""
    return shapley_whittle_iterative_policy(env,state,budget,lamb,memory,per_epoch_results,contextual=False)


def contextual_whittle_policy(env,state,budget,lamb,memory,per_epoch_results):
    """Contextual Linear-Whittle Policy; basically consider the reward when
    you know the context in the current timestep, and estimate for future timesteps
    through an expansion of the state space
    
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
    num_samples = 10

    if memory == None:
        # Construct transitions of size |S|*|D|, where we get |D| samples

        random_contexts = np.array([env.get_random_context() for _ in range(num_samples)])
        new_reward_matrix = np.zeros((N,num_samples*n_states+1,2))
        transitions = np.zeros((N,num_samples*n_states+1,2,num_samples*n_states+1))

        context_shapley_values = np.zeros((N,num_samples*n_states+1))

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    default_state = [env.worst_state for _ in range(N)]
                    default_state[i] = j
                    context = random_contexts[k]
                    res = contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.active_states,context)
                    context_shapley_values[i,idx] = res
        
            default_state = [env.worst_state for _ in range(N)]
            default_state[i] = state[i]
            context_shapley_values[i,-1] = contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.active_states,env.context)

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    if j in env.active_states:
                        new_reward_matrix[i,idx] += lamb/N
                    default_state = [env.worst_state for _ in range(N)]
                    default_state[i] = j
                    new_reward_matrix[i,idx,1] += (1-lamb)*context_shapley_values[i,j*num_samples+k]
                    transitions[i][idx][:,:-1] = np.repeat(env.transitions[i][j], num_samples,axis=1)
                    transitions[i][idx] /= num_samples 

            if state[i] in env.active_states:
                new_reward_matrix[i,-1] += lamb/N

            transitions[i][-1] = transitions[i][state[i]*num_samples]
            new_reward_matrix[i,-1,1] += (1-lamb)*context_shapley_values[i,-1]
    else:
        new_reward_matrix, transitions = memory 

        contextual_shapley_values = []
        for i in range(N):
            default_state = [env.worst_state for _ in range(N)]
            default_state[i] = state[i]
            contextual_shapley_values.append(contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.active_states,env.context))

        for i in range(N):
            transitions[i][-1] = transitions[i][state[i]*num_samples]
            new_reward_matrix[i,-1] = 0

            if state[i] in env.active_states:
                new_reward_matrix[i,-1] += lamb/N 
            new_reward_matrix[i,-1,1] += (1-lamb)*contextual_shapley_values[i]

    state_WI = []
    for i in range(N):        
        better_reward = deepcopy(new_reward_matrix[i])
        state_WI_value = fast_compute_whittle_indices(transitions[i],better_reward,env.discount)
        state_WI.append(state_WI_value[-1])

    sorted_WI = np.argsort(state_WI)[::-1]
    filtered_WI = [i for i in sorted_WI if state_WI[i] >= 0]

    action = np.zeros(N, dtype=np.int8)
    action[filtered_WI[:budget]] = 1

    return action, (new_reward_matrix,transitions)  

def fast_contextual_whittle_policy(env,state,budget,lamb,memory,per_epoch_results):
    """Faster version of the contextual_whittle_policy, which is faster by sampling 
    fewer contexts, and using only one context (the average)
    This is less accurate however
    
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
    num_samples = 1

    if memory == None:
        # Construct transitions of size |S|*|D|, where we get |D| samples

        random_contexts = np.array(env.match_probability_list[[i//env.volunteers_per_arm for i in env.agent_idx]])
        new_reward_matrix = np.zeros((N,num_samples*n_states+1,2))
        transitions = np.zeros((N,num_samples*n_states+1,2,num_samples*n_states+1))

        context_shapley_values = np.zeros((N,num_samples*n_states+1))

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    default_state = [env.worst_state for _ in range(N)]
                    default_state[i] = j
                    context = random_contexts[k]
                    res = contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[[i//env.volunteers_per_arm for i in env.agent_idx]],env.reward_type,env.reward_parameters,env.active_states,context)
                    context_shapley_values[i,idx] = res
        
            default_state = [env.worst_state for _ in range(N)]
            default_state[i] = state[i]
            context_shapley_values[i,-1] = contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[[i//env.volunteers_per_arm for i in env.agent_idx]],env.reward_type,env.reward_parameters,env.active_states,env.context)

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    if j in env.active_states:
                        new_reward_matrix[i,idx] += lamb/N
                    default_state = [env.worst_state for _ in range(N)]
                    default_state[i] = j
                    new_reward_matrix[i,idx,1] += (1-lamb)*context_shapley_values[i,j*num_samples+k]
                    transitions[i][idx][:,:-1] = np.repeat(env.transitions[i][j], num_samples,axis=1)
                    transitions[i][idx] /= num_samples 

            if state[i] in env.active_states:
                new_reward_matrix[i,-1] += lamb/N

            transitions[i][-1] = transitions[i][state[i]*num_samples]
            new_reward_matrix[i,-1,1] += (1-lamb)*context_shapley_values[i,-1]
    else:
        new_reward_matrix, transitions = memory 

        contextual_shapley_values = []
        for i in range(N):
            default_state = [env.worst_state for _ in range(N)]
            default_state[i] = state[i]
            contextual_shapley_values.append(contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.active_states,env.context))

        for i in range(N):
            transitions[i][-1] = transitions[i][state[i]*num_samples]
            new_reward_matrix[i,-1] = 0

            if state[i] in env.active_states:
                new_reward_matrix[i,-1] += lamb/N 
            new_reward_matrix[i,-1,1] += (1-lamb)*contextual_shapley_values[i]

    state_WI = []
    for i in range(N):        
        better_reward = deepcopy(new_reward_matrix[i])
        state_WI_value = fast_compute_whittle_indices(transitions[i],better_reward,env.discount)
        state_WI.append(state_WI_value[-1])

    sorted_WI = np.argsort(state_WI)[::-1]
    filtered_WI = [i for i in sorted_WI if state_WI[i] >= 0]

    action = np.zeros(N, dtype=np.int8)
    action[filtered_WI[:budget]] = 1

    return action, (new_reward_matrix,transitions)  


def contextual_shapley_policy(env,state,budget,lamb,memory,per_epoch_results):
    """Contextual Shapley policy, which computes arm values by expanding the 
    state space, and sampling different contexts
    For a faster version, see fast_contextual_shapley_policy
    
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

    if memory == None:
        # Construct transitions of size |S|*|D|, where we get |D| samples
        num_samples = 10

        random_contexts = np.array([env.get_random_context()for _ in range(num_samples)])
        new_reward_matrix = np.zeros((N,num_samples*n_states+1,2))
        transitions = np.zeros((N,num_samples*n_states+1,2,num_samples*n_states+1))

        context_shapley_values = np.zeros((N,num_samples*n_states+1))

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    default_state = [env.best_state for _ in range(N)]
                    default_state[i] = j
                    context = random_contexts[k]
                    res = shapley_index_custom_contexts(env,default_state,context,idx=i)
                    context_shapley_values[i,idx] = res
        
            default_state = [env.best_state for _ in range(N)]
            default_state[i] = state[i]
            context_shapley_values[i,-1] = shapley_index_custom_contexts(env,default_state,env.context,idx=i)

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    if j in env.active_states:
                        new_reward_matrix[i,idx] += lamb/N
                    default_state = [env.worst_state for _ in range(N)]
                    default_state[i] = j
                    new_reward_matrix[i,idx,1] += (1-lamb)*context_shapley_values[i,j*num_samples+k]
                    transitions[i][idx][:,:-1] = np.repeat(env.transitions[i][j], num_samples,axis=1)
                    transitions[i][idx] /= num_samples 

            if state[i] in env.active_states:
                new_reward_matrix[i,-1] += lamb/N

            transitions[i][-1] = transitions[i][state[i]*num_samples]
            new_reward_matrix[i,-1,1] += (1-lamb)*context_shapley_values[i,-1]
    else:
        new_reward_matrix, transitions = memory 
        num_samples = 10

        contextual_shapley_values = []
        for i in range(N):
            default_state = [env.best_state for _ in range(N)]
            default_state[i] = state[i]
            contextual_shapley_values.append(shapley_index_custom_contexts(env,default_state,env.context,idx=i))

        for i in range(N):
            transitions[i][-1] = transitions[i][state[i]*num_samples]
            new_reward_matrix[i,-1] = 0

            if state[i] in env.active_states:
                new_reward_matrix[i,-1] += lamb/N 
            new_reward_matrix[i,-1,1] += (1-lamb)*contextual_shapley_values[i]

    state_WI = []
    for i in range(N):        
        better_reward = deepcopy(new_reward_matrix[i])
        state_WI_value = fast_compute_whittle_indices(transitions[i],better_reward,env.discount)


        state_WI.append(state_WI_value[-1])

    sorted_WI = np.argsort(state_WI)[::-1]
    filtered_WI = [i for i in sorted_WI if state_WI[i] >= 0]
    action = np.zeros(N, dtype=np.int8)
    action[filtered_WI[:budget]] = 1

    return action, (new_reward_matrix,transitions) 

def fast_contextual_shapley_policy(env,state,budget,lamb,memory,per_epoch_results):
    """Run a faster version of the contextual Shapley Policy
    Essentially use one sample (which is the average), which reduces the 
    State space needed to compute things
    
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
    start = time.time()

    if memory == None:
        start = time.time()
        # Construct transitions of size |S|*|D|, where we get |D| samples
        num_samples = 1

        random_contexts = np.array([env.match_probability_list[env.agent_idx] for _ in range(num_samples)])
        new_reward_matrix = np.zeros((N,num_samples*n_states+1,2))
        transitions = np.zeros((N,num_samples*n_states+1,2,num_samples*n_states+1))

        context_shapley_values = np.zeros((N,num_samples*n_states+1))

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    default_state = [env.best_state for _ in range(N)]
                    default_state[i] = j
                    context = random_contexts[k]
                    res = shapley_index_custom_contexts(env,default_state,context,idx=i)
                    context_shapley_values[i,idx] = res
        
            default_state = [env.best_state for _ in range(N)]
            default_state[i] = state[i]
            context_shapley_values[i,-1] = shapley_index_custom_contexts(env,default_state,env.context,idx=i)

        for i in range(N):
            for j in range(n_states):
                for k in range(num_samples):
                    idx = j*num_samples + k 
                    if j in env.active_states:
                        new_reward_matrix[i,idx] += lamb/N
                    default_state = [env.worst_state for _ in range(N)]
                    default_state[i] = j
                    new_reward_matrix[i,idx,1] += (1-lamb)*context_shapley_values[i,j*num_samples+k]
                    transitions[i][idx][:,:-1] = np.repeat(env.transitions[i][j], num_samples,axis=1)
                    transitions[i][idx] /= num_samples 

            if state[i] in env.active_states:
                new_reward_matrix[i,-1] += lamb/N

            transitions[i][-1] = transitions[i][state[i]*num_samples]
            new_reward_matrix[i,-1,1] += (1-lamb)*context_shapley_values[i,-1]

    else:
        new_reward_matrix, transitions = memory 
        num_samples = 1

        contextual_shapley_values = []
        default_state = [env.best_state for _ in range(N)]
        for i in range(N):
            default_state[i] = state[i]
            contextual_shapley_values.append(shapley_index_custom_contexts(env,default_state,env.context,idx=i))
            default_state[i] = env.best_state

        for i in range(N):
            transitions[i][-1] = transitions[i][state[i]*num_samples]
            new_reward_matrix[i,-1] = 0

            if state[i] in env.active_states:
                new_reward_matrix[i,-1] += lamb/N 
            new_reward_matrix[i,-1,1] += (1-lamb)*contextual_shapley_values[i]

    state_WI = []
    for i in range(N):        
        better_reward = deepcopy(new_reward_matrix[i])
        state_WI_value = fast_compute_whittle_indices(transitions[i],better_reward,env.discount)

        state_WI.append(state_WI_value[-1])

    sorted_WI = np.argsort(state_WI)[::-1]
    filtered_WI = [i for i in sorted_WI if state_WI[i] >= 0]
    action = np.zeros(N, dtype=np.int8)
    action[filtered_WI[:budget]] = 1

    return action, (new_reward_matrix,transitions) 