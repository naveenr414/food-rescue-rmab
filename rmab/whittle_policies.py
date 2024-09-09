""" Index-Based Algorithms for matching, activity """

import numpy as np

from rmab.utils import Memoizer, contextual_custom_reward, binary_to_decimal, custom_reward, one_hot, one_hot_fixed, shapley_index_custom, shapley_index_custom_contexts, shapley_index_custom_fixed, compute_u_matrix
from rmab.compute_whittle import arm_value_iteration_exponential, fast_arm_compute_whittle, fast_arm_compute_whittle_multi_prob, arm_compute_whittle, arm_compute_whittle_multi_prob, fast_compute_whittle_indices
from rmab.baseline_policies import compute_reward_matrix, compute_p_matrix

from copy import deepcopy
import random 



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
        reward_matrix = compute_reward_matrix(env,N,lamb)
        whittle_matrix = np.zeros((N,n_states))
        for i in range(N):
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
        p_matrix = compute_p_matrix(env,N)
        reward_matrix = compute_reward_matrix(env,N,lamb)
        reward_matrix[:,:,1] += (1-lamb)*p_matrix
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
    
    state_WI = [whittle_matrix[i][state[i]] for i in range(N)]

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (whittle_matrix)  

def whittle_match_prob_now(env,state,N,pulled_arms):
    if len(pulled_arms) > 0:
        pulled_action = one_hot_fixed(pulled_arms[0],len(state),pulled_arms)
        default_custom_reward = contextual_custom_reward(state,pulled_action,np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.context)
    else:
        pulled_action = [0 for i in range(len(state))]
        default_custom_reward = 0
    
    match_prob_all = []
    for i in range(N):
        new_action = deepcopy(pulled_action)
        new_action[i] = 1

        match_prob_all.append(contextual_custom_reward(state,new_action,np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.context)-default_custom_reward)
    return match_prob_all

def shapley_match_prob_now(env,state,N,pulled_arms):
    match_prob_all = np.array(shapley_index_custom_fixed(env,state,{},pulled_arms,env.context)[0])
    return match_prob_all

def iterative_policy_skeleton(env,state,budget,lamb,memory,per_epoch_results,match_prob_now_function,reward_matrix):
    N = len(state)
    n_states = env.transitions.shape[1]
    
    action = np.zeros(N, dtype=np.int8)
    pulled_arms = []

    for _ in range(budget):
        match_prob_now_list = match_prob_now_function(env,state,N,pulled_arms)
        state_WI = []

        for i in range(N):
            if i not in pulled_arms:
                match_prob_now = match_prob_now_list[i]
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

    if memory is None:
        p_matrix = compute_p_matrix(env,N)
        reward_matrix = compute_reward_matrix(env,N,lamb)
        reward_matrix[:,:,1] += (1-lamb)*p_matrix
    else:
        reward_matrix = memory 

    return iterative_policy_skeleton(env,state,budget,lamb,memory,per_epoch_results,whittle_match_prob_now,reward_matrix)

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
        u_matrix = compute_u_matrix(env,N,n_states)
        reward_matrix = compute_reward_matrix(env,N,lamb)
        reward_matrix[:,:,1] += (1-lamb)*u_matrix
    else:
        reward_matrix = memory 

    return iterative_policy_skeleton(env,state,budget,lamb,memory,per_epoch_results,shapley_match_prob_now,reward_matrix)

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
    num_samples = 25

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
                    res = contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,context)
                    context_shapley_values[i,idx] = res
        
            default_state = [env.worst_state for _ in range(N)]
            default_state[i] = state[i]
            context_shapley_values[i,-1] = contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.context)

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
            contextual_shapley_values.append(contextual_custom_reward(default_state,one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.context))

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
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (new_reward_matrix,transitions)  

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
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (new_reward_matrix,transitions) 