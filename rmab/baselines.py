""" Oracle algorithms for matching, activity """

import numpy as np
import heapq

from rmab.uc_whittle import Memoizer
from rmab.compute_whittle import arm_compute_whittle, arm_value_iteration_exponential, arm_value_iteration_neural, arm_compute_whittle_sufficient
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

def optimal_whittle(env, n_episodes, n_epochs, discount,reward_function='activity',lamb=0):
    """Whittle index policy based on computing the subsidy for each arm
    This approximates the problem as the sum of Linear rewards, then 
    Decomposes the problem into the problem for each arm individually
    
    reward_function: String, either
        activity: Maximize the total activity, s_i 
        matching: Maximize the total number of matches, s_i*a_i 
        combined: Maximize s_i*a_i + \lambda s_i
    lamb: Float, hyperparameter for the combined matching """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    memoizer = Memoizer('optimal')

    all_reward = np.zeros((n_epochs, T))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        true_transitions = env.transitions

        match_prob = env.match_probability
        if env.match_probability_list == []:
            match_probability_list = np.array([match_prob for i in range(N)])
        else:
            match_probability_list = np.array(env.match_probability_list)[env.cohort_idx]

        print('first state', env.observe())

        for t in range(0, T):
            state = env.observe()

            # select optimal action based on known transition probabilities
            # compute whittle index for each arm
            state_WI = np.zeros(N)
            top_WI = []
            min_chosen_subsidy = -1 #0
            for i in range(N):
                arm_transitions = true_transitions[i, :, :, 1]

                # memoize to speed up
                check_set_val = memoizer.check_set(arm_transitions, state[i])
                if check_set_val != -1:
                    state_WI[i] = check_set_val
                else:
                    state_WI[i] = arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=match_probability_list[i])
                    memoizer.add_set(arm_transitions, state[i], state_WI[i])

                if len(top_WI) < budget:
                    heapq.heappush(top_WI, (state_WI[i], i))
                else:
                    # add state_WI to heap
                    heapq.heappushpop(top_WI, (state_WI[i], i))
                    min_chosen_subsidy = top_WI[0][0]  # smallest-valued item

            # pull K highest indices
            sorted_WI = np.argsort(state_WI)[::-1]
            print(f'   state {state} state_WI {np.round(state_WI, 2)} sorted {np.round(sorted_WI[:budget], 2)}')

            action = np.zeros(N, dtype=np.int8)
            action[sorted_WI[:budget]] = 1

            _, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward

    return all_reward

def optimal_q_iteration(env, n_episodes, n_epochs, discount,reward_function='matching',lamb=0):
    """Q-iteration to solve which arms to pull
        Doesn't decompose the arms, but can achieve higher performance
        Is quite slow 

    More details in arm_value_iteration_match function
     """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes
    match_prob = env.match_probability

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        true_transitions = env.transitions

        print('first state', env.observe())

        Q_vals = arm_value_iteration_exponential(true_transitions,discount,budget,match_prob,
                        reward_function=reward_function,lamb=lamb,
                        match_probability_list=np.array(
                            env.match_probability_list)[env.cohort_idx])

        for t in range(0, T):
            state = env.observe()
            state_rep = binary_to_decimal(state)

            max_action = np.argmax(Q_vals[state_rep])
            binary_val = bin(max_action)[2:].zfill(N)

            action = np.zeros(N, dtype=np.int8)
            action = np.array([int(i) for i in binary_val])

            assert np.sum(action) == budget 

            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward


    return all_reward

def optimal_whittle_sufficient(env, n_episodes, n_epochs, discount,reward_function='activity',lamb=0):
    """Whittle index policy based on computing the subsidy for each arm
    This approximates the problem as the sum of Linear rewards, then 
    Decomposes the problem into the problem for each arm individually
    
    reward_function: String, either
        activity: Maximize the total activity, s_i 
        matching: Maximize the total number of matches, s_i*a_i 
        combined: Maximize s_i*a_i + \lambda s_i
    lamb: Float, hyperparameter for the combined matching """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes
    match_prob = env.match_probability

    env.reset_all()

    memoizer = Memoizer('optimal')

    all_reward = np.zeros((n_epochs, T))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        true_transitions = env.transitions
        print('first state', env.observe())
        
        p_0_0_1 = 0.3 # np.mean(true_transitions[:,0,0,1])
        p_1_1_1 = 0.7 # np.mean(true_transitions[:,1,1,1])
        p_1_0_1 = 0.6 # np.mean(true_transitions[:,1,0,1])

        binom_pmfs = [[0 for i in range(N+1)] for i in range(N+1)]

        for num_1 in range(0,N+1):
            num_0 = N-num_1 

            for i_0_0_1 in range(num_0+1):
                for i_1_1_1 in range(0,min(num_1,budget)+1):
                    for i_1_0_1 in range(0,num_1-i_1_1_1+1):
                        p_a = binom.pmf(i_0_0_1,num_0,p_0_0_1)
                        p_b = binom.pmf(i_1_1_1,min(num_1,budget),p_1_1_1)
                        p_c = binom.pmf(i_1_0_1,num_1-i_1_1_1,p_1_0_1)
                        binom_pmfs[num_1][i_0_0_1+i_1_1_1+i_1_0_1] += p_a*p_b*p_c 
        
        # for i in range(len(a_0)):
        #     for j in range(len(a_1)):
        #         for k in range(len(a_2)):

        #         binom_pmfs[i+j] += a_0[i]*a_1[j]
        
        for t in range(0, T):
            state = env.observe()
            state_str = [str(i) for i in state]
            T_stat = np.sum(state)

            # select optimal action based on known transition probabilities
            # compute whittle index for each arm
            state_WI = np.zeros(N)
            top_WI = []
            min_chosen_subsidy = -1 #0

            for i in range(N):
                arm_transitions = true_transitions[i, :, :, 1]

                # memoize to speed up
                check_set_val = memoizer.check_set(arm_transitions, str(state[i])+ ' '+str(T_stat))
                if check_set_val != -1:
                    state_WI[i] = check_set_val
                else:
                    state_WI[i] = arm_compute_whittle_sufficient(arm_transitions, state[i], T_stat,discount, min_chosen_subsidy,N,match_prob,budget,reward_function=reward_function,lamb=lamb,probs=binom_pmfs)
                    memoizer.add_set(arm_transitions, str(state[i])+' '+str(T_stat), state_WI[i])

                if len(top_WI) < budget:
                    heapq.heappush(top_WI, (state_WI[i], i))
                else:
                    # add state_WI to heap
                    heapq.heappushpop(top_WI, (state_WI[i], i))
                    min_chosen_subsidy = top_WI[0][0]  # smallest-valued item

            # pull K highest indices
            sorted_WI = np.argsort(state_WI)[::-1]

            action = np.zeros(N, dtype=np.int8)
            action[sorted_WI[:budget]] = 1

            _, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward

    return all_reward

def optimal_neural_q_iteration(env, budget, match_prob,n_episodes, n_epochs, discount,reward_function='matching',lamb=0):
    """Q-iteration to solve which arms to pull
        Doesn't decompose the arms, but can achieve higher performance
        Is quite slow 

    More details in arm_value_iteration_match function
     """
    T         = env.episode_len * n_episodes

    obs, info = env.reset(options={'reset_type': 'full'})

    all_reward = np.zeros((n_epochs, T))
    total_active = 0 

    for epoch in range(n_epochs):
        if epoch != 0: 
            env.reset(options={'reset_type': 'instance'})

        print('first state', obs)

        net = Net(env,device='cuda')
        agent = Agent(net)
        agent.train(total_time_steps=1000)
        obs, info = env.reset(options={'reset_type': 'to_0'})

        for t in range(0, T):
            # select optimal action based on known transition probabilities
            # compute whittle index for each arm
            action, _ = agent.act(obs)

            obs, reward, done, total_active_temp = env.step(action)
            
            if np.sum(action)<=budget:
                reward = 1-np.power(1-match_prob,obs.flatten().dot(action.flatten()))

            else:
                reward = 0

            # if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward
        total_active += total_active_temp[0]['final_info']['total_active']

    return all_reward, total_active/(all_reward.size*action.size)

def random_policy(env, n_episodes, n_epochs):
    """ random action each timestep """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        print('first state', env.observe())

        for t in range(0, T):
            state = env.observe()

            # select arms at random
            selected_idx = np.random.choice(N, size=budget, replace=False)
            action = np.zeros(N, dtype=np.int8)
            action[selected_idx] = 1

            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward

    return all_reward

def greedy_policy(env, n_episodes, n_epochs,discount,reward_function='matching',lamb=0):
    """ random action each timestep """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    if reward_function != 'combined':
        raise Exception("Reward function is not matching; greedy is designed for match + activity")

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        print('first state', env.observe())

        for t in range(0, T):
            state = env.observe()

            score_by_agent = [0 for i in range(N)]
            true_transitions = env.transitions
            match_probabilities = np.array(env.match_probability_list)[env.cohort_idx]

            for i in range(N):
                activity_score = true_transitions[i,state[i],1,1]
                activity_score -= true_transitions[i,state[i],0,1]
                
                matching_score = state[i]*match_probabilities[i]
                score_by_agent[i] = matching_score + activity_score * lamb 

            # select arms at random
            selected_idx = np.argsort(score_by_agent)[-budget:][::-1]
            action = np.zeros(N, dtype=np.int8)
            action[selected_idx] = 1

            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward

    return all_reward

def greedy_iterative_policy(env, n_episodes, n_epochs,discount,reward_function='matching',lamb=0,use_Q=False,use_whittle=False,use_shapley=False):
    """ random action each timestep """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    if reward_function != 'combined':
        raise Exception("Reward function is not matching; greedy is designed for match + activity")

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        print('first state', env.observe())
        match_probabilities = np.array(env.match_probability_list)[env.cohort_idx]
        true_transitions = env.transitions

        if use_Q:
            Q_vals = arm_value_iteration_exponential(env.transitions, discount, budget, env.match_probability,reward_function=reward_function,lamb=lamb,match_probability_list=match_probabilities)

        reward_by_group = np.zeros((2**N,2**N))
        for temp_state in list(itertools.product([0, 1], repeat=N)):
            for bitstring in list(itertools.product([0, 1], repeat=N)):
                activity_score = 0
                non_match_prob = 1

                for i in range(len(bitstring)):
                    if bitstring[i] == 1:
                        activity_score += true_transitions[i,temp_state[i],1,1]
                        activity_score -= true_transitions[i,temp_state[i],0,1]
                        non_match_prob *= (1-match_probabilities[i]*temp_state[i])
                reward_by_group[binary_to_decimal(temp_state),binary_to_decimal(bitstring)] = non_match_prob + lamb*activity_score

        for t in range(0, T):
            state = env.observe()

            selected_idx = []
            current_non_match_prob = 1

            score_by_agent = [0 for i in range(N)]
            match_probabilities = np.array(env.match_probability_list)[env.cohort_idx]

            for _ in range(budget):
                current_action = [0 for i in range(N)]
                for i in selected_idx:
                    current_action[i] = 1 
                encoded_state = binary_to_decimal(state)

                scores = []
                for i in range(N):
                    if i in selected_idx:
                        continue 
                    
                    if use_whittle:
                        arm_transitions = true_transitions[i, :, :, 1]
                        min_chosen_subsidy = -1
                        score = arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=(1-current_non_match_prob*(1-match_probabilities[i])))
                    else:
                        if use_shapley:
                            score = 0
                            num = 0
                            non_selected = [j for j in range(N) if j not in selected_idx and j!=i]
                            for combo in powerset(non_selected):
                                indices = [0 for i in range(N)]
                                for j in combo:
                                    indices[j] = 1 
                                for j in selected_idx:
                                    indices[j] = 1 
                                initial_score = reward_by_group[binary_to_decimal(state)][binary_to_decimal(indices)]
                                indices[i] = 1
                                final_score = reward_by_group[binary_to_decimal(state)][binary_to_decimal(indices)]
                                num += 1
                                score += (final_score-initial_score)
                            score /= num 

                        else:
                            activity_score = true_transitions[i,state[i],1,1]
                            activity_score -= true_transitions[i,state[i],0,1]
                            
                            matching_score = current_non_match_prob - current_non_match_prob*(1-match_probabilities[i]*state[i])                            
                            score = matching_score + activity_score * lamb 
                        if use_Q:
                            new_action = deepcopy(current_action)
                            new_action[i] = 1

                            encoded_action = binary_to_decimal(new_action)
                            future_score = Q_vals[encoded_state,encoded_action]
                            score += discount*future_score
                    scores.append((score,i))
                selected_idx.append(max(scores,key=lambda k: k[0])[1])
                current_non_match_prob *= (1-match_probabilities[selected_idx[-1]])

            # select arms at random
            selected_idx = np.array(selected_idx)
            action = np.zeros(N, dtype=np.int8)
            action[selected_idx] = 1

            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward

    return all_reward

def mcts_policy(env, n_episodes, n_epochs,discount,reward_function='matching',lamb=0,use_Q=False,use_whittle=False,use_shapley=False):
    """ random action each timestep """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    if reward_function != 'combined':
        raise Exception("Reward function is not matching; greedy is designed for match + activity")

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        print('first state', env.observe())
        match_probabilities = np.array(env.match_probability_list)[env.cohort_idx]
        true_transitions = env.transitions

        if use_Q:
            Q_vals = arm_value_iteration_exponential(env.transitions, discount, budget, env.match_probability,reward_function=reward_function,lamb=lamb,match_probability_list=match_probabilities)

        reward_by_group = np.zeros((2**N,2**N))
        for temp_state in list(itertools.product([0, 1], repeat=N)):
            for bitstring in list(itertools.product([0, 1], repeat=N)):
                activity_score = 0
                non_match_prob = 1

                for i in range(len(bitstring)):
                    if bitstring[i] == 1:
                        activity_score += true_transitions[i,temp_state[i],1,1]
                        activity_score -= true_transitions[i,temp_state[i],0,1]
                        non_match_prob *= (1-match_probabilities[i]*temp_state[i])
                reward_by_group[binary_to_decimal(temp_state),binary_to_decimal(bitstring)] = non_match_prob + lamb*activity_score

        for t in range(0, T):
            state = env.observe()

            selected_idx = []
            current_non_match_prob = 1

            score_by_agent = [0 for i in range(N)]
            match_probabilities = np.array(env.match_probability_list)[env.cohort_idx]

            for _ in range(budget):
                current_action = [0 for i in range(N)]
                for i in selected_idx:
                    current_action[i] = 1 
                encoded_state = binary_to_decimal(state)

                scores = []
                for i in range(N):
                    if i in selected_idx:
                        continue 
                    
                    if use_whittle:
                        arm_transitions = true_transitions[i, :, :, 1]
                        min_chosen_subsidy = -1
                        score = arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=(1-current_non_match_prob*(1-match_probabilities[i])))
                    else:
                        if use_shapley:
                            score = 0
                            num = 0
                            non_selected = [j for j in range(N) if j not in selected_idx and j!=i]
                            for combo in powerset(non_selected):
                                indices = [0 for i in range(N)]
                                for j in combo:
                                    indices[j] = 1 
                                for j in selected_idx:
                                    indices[j] = 1 
                                initial_score = reward_by_group[binary_to_decimal(state)][binary_to_decimal(indices)]
                                indices[i] = 1
                                final_score = reward_by_group[binary_to_decimal(state)][binary_to_decimal(indices)]
                                num += 1
                                score += (final_score-initial_score)
                            score /= num 

                        else:
                            activity_score = true_transitions[i,state[i],1,1]
                            activity_score -= true_transitions[i,state[i],0,1]
                            
                            matching_score = current_non_match_prob - current_non_match_prob*(1-match_probabilities[i]*state[i])                            
                            score = matching_score + activity_score * lamb 
                        if use_Q:
                            new_action = deepcopy(current_action)
                            new_action[i] = 1

                            encoded_action = binary_to_decimal(new_action)
                            future_score = Q_vals[encoded_state,encoded_action]
                            score += discount*future_score
                    scores.append((score,i))
                selected_idx.append(max(scores,key=lambda k: k[0])[1])
                current_non_match_prob *= (1-match_probabilities[selected_idx[-1]])

            # select arms at random
            selected_idx = np.array(selected_idx)
            action = np.zeros(N, dtype=np.int8)
            action[selected_idx] = 1

            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward

    return all_reward

def WIQL(env, n_episodes, n_epochs):
    """ Whittle index-based Q-Learning
    [Biswas et al. 2021]

    input: N, budget, alpha(c), initial states
    """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    def alpha_func(c):
        """ learning parameter
        alpha = 0: agent doesn't learn anything new
        alpha = 1: stores only most recent information; overwrites previous records """
        assert 0 <= c <= 1
        return c

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()

        # initialize
        Q_vals    = np.zeros((N, n_states, n_actions))
        lamb_vals = np.zeros((N, n_states))

        print('first state', env.observe())
        for t in range(0, T):
            state = env.observe()

            # select M arms using epsilon-decay policy
            epsilon = N / (N + t)

            # with probability epsilon, select B arms uniformly at random
            if np.random.binomial(1, epsilon):
                selected_idx = np.random.choice(N, size=budget, replace=False)
            else:
                state_lamb_vals = np.array([lamb_vals[i, state[i]] for i in range(N)])
                # select top arms according to their lambda values
                selected_idx = np.argpartition(state_lamb_vals, -budget)[-budget:]
                selected_idx = selected_idx[np.argsort(state_lamb_vals[selected_idx])][::-1] # sort indices

            action = np.zeros(N, dtype=np.int8)
            action[selected_idx] = 1

            # take suitable actions on arms
            # execute chosen policy; observe reward and next state
            next_state, reward, done, _ = env.step(action)
            if done and t+1 < T: env.reset()

            # update Q, lambda
            c = .5 # None
            for i in range(N):
                for s in range(n_states):
                    for a in range(n_actions):
                        prev_Q = Q_vals[i, s, a]
                        state_i = next_state[i]
                        prev_max_Q = np.max(Q_vals[i, state_i, :])

                        alpha = alpha_func(c)

                        Q_vals[i, s, a] = (1 - alpha) * prev_Q + alpha * (reward + prev_max_Q)

                    lamb_vals[i, s] = Q_vals[i, s, 1] - Q_vals[i, s, 0]

            all_reward[epoch, t] = reward

    return all_reward

def myopic_match_n_step(env, n_episodes, n_epochs, discount,n_step):
    """Compute the greedy policy for matching, alerting those who are in 
    the good state n_steps in the future first, then alert those n_steps+1 next, etc. 
    Arguments
        env: RMAB simulator
        n_episodes: Integer, number of episodes
        n_epochs: Integer, number of epochs
        discount: Float, gamma
        n_steps: Integer, number of steps to lookahead
            If 0, then we alert those in s=1
            If 1, we predict who will be in s=1 next
            If s=-1, then we use the steady state (s=\infinity)
            Otherwise, we use transition_matrix^n_steps
    Returns: Rewards for each epoch/episode"""
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        
        if n_step == 0 or n_step == 1:
            power_transitions = [t[:,1,:] for t in env.transitions]
        # Look ahead more than 1 step 
        elif n_step > 1:
            power_transitions = [np.linalg.matrix_power(t[:,1,:],n_step) for t in env.transitions] 
        # Look ahead infinite steps (steady state)
        elif n_step == -1:
            stationary_distribution = [get_stationary_distribution(t[:,1,:]) for t in env.transitions]
            power_transitions = [[r,r] for r in stationary_distribution]
 
        print('first state', env.observe())

        for t in range(0, T):
            state = env.observe()

            # select optimal action based on known transition probabilities
            # compute whittle index for each arm

            greedy_transitions = [i for i in range(len(state)) if state[i] == 1]
            lookahead_transitions = [(i,power_transitions[i][state[i]][1]) for i in range(len(state))]

            lookahead_transitions = sorted(lookahead_transitions,key=lambda k: k[1],reverse=True)

            if n_step != 0:
                indices = lookahead_transitions[:budget]
                indices = [i[0] for i in indices]
            else:
                indices = greedy_transitions 
                if len(indices) > budget:
                    indices = np.random.choice(indices,budget)
                elif len(indices) < budget:
                    others = [i[0] for i in lookahead_transitions if i[0] not in indices]
                    indices += others[:budget-len(indices)]

            action = np.zeros(N, dtype=np.int8)
            action[indices] = 1
            _, reward, done, _ = env.step(action)
            if done and t+1 < T: env.reset()
            all_reward[epoch, t] = reward

    return all_reward




