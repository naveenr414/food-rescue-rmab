""" Oracle algorithms for matching, activity """

import numpy as np
import heapq

from rmab.uc_whittle import Memoizer
from rmab.compute_whittle import arm_compute_whittle, arm_value_iteration_exponential, arm_value_iteration_approximate, arm_value_iteration_neural
from rmab.utils import get_stationary_distribution, binary_to_decimal, list_to_binary
from itertools import combinations

import torch
import torch.nn as nn
import torch.optim as optim


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
                    state_WI[i] = arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb)
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

        Q_vals = arm_value_iteration_exponential(true_transitions,discount,budget,match_prob,reward_function=reward_function,lamb=lamb)

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

def optimal_neural_q_iteration(env, n_episodes, n_epochs, discount,reward_function='matching',lamb=0):
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

        Q_network = arm_value_iteration_neural(true_transitions,discount,budget,match_prob,reward_function=reward_function,lamb=lamb)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        
        for t in range(0, T):
            state = env.observe()
            state_rep = torch.Tensor(state).to(device)

            max_action = torch.argmax(Q_network(state_rep).cpu()).item()
            binary_val = bin(max_action)[2:].zfill(N)

            action = np.zeros(N, dtype=np.int8)
            action = np.array([int(i) for i in binary_val])
            assert np.sum(action) <= budget 

            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward


    return all_reward

def optimal_sufficient_q(env, n_episodes, n_epochs, discount,reward_function='matching',lamb=0):
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

        Q_vals_list = [arm_value_iteration_approximate(true_transitions,discount,budget,match_prob,reward_function=reward_function,lamb=lamb,arm_num=i) for i in range(N)]
        all_a = list(combinations(range(N), budget))
        all_a = [np.array(list_to_binary(i,N)) for i in all_a]

        for t in range(0, T):
            state = env.observe()
            state_rep = binary_to_decimal(state)

            rewards = []
            best_action = -1

            for action in all_a:
                total_reward = 0 
                for arm_num in range(N):
                    s_ = state[arm_num] 
                    a_ = action[arm_num] 
                    all_actions = binary_to_decimal(action)
                    S_ = np.dot(state,action) 
                    T_ = np.sum([true_transitions[i][state[i]][action[i]][1] for i in range(N)])
                    T_ = int(round(T_))                    
                    total_reward += Q_vals_list[arm_num][s_,a_,all_actions,S_,T_]

                if best_action == -1 or rewards[best_action]<total_reward:
                    best_action = len(rewards)
                rewards.append(total_reward)

            action = all_a[best_action]

            assert np.sum(action) == budget 

            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward


    return all_reward


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




