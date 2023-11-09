""" baseline algorithms """

import numpy as np
import random
import heapq

from rmab.simulator import RMABSimulator, random_valid_transition
from rmab.uc_whittle import Memoizer
from rmab.compute_whittle import arm_compute_whittle, arm_value_iteration_match, binary_to_decimal
from rmab.utils import get_stationary_distribution

def approximate_match_whittle(env, n_episodes, n_epochs, discount,use_match_reward=False):
    def solve_gamma_2(T,gamma,s):
        a = T[0,0,0]
        b = T[0,0,1]
        c = T[0,1,0]
        d = T[0,1,1]
        e = T[1,0,0]
        f = T[1,0,1]
        g = T[1,1,0]
        h = T[1,1,1]

        if s == 0:
            top = gamma*(-b+d+b*c*gamma-a*d*gamma)
            bottom = (1-a*gamma-b*gamma+d*gamma-h*gamma+b*c*gamma**2-a*d*gamma**2-b*g*gamma**2+a*h*gamma**2)

            return top/bottom 
        else: 
            top = -1+c*gamma+f*gamma+d*e*gamma**2-c*f*gamma**2
            top*=-1 
            bottom = 1-c*gamma-e*gamma-f*gamma+g*gamma-d*e*gamma**2+c*f*gamma**2-f*g*gamma**2+e*h*gamma**2

            return top/bottom 

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

            # select optimal action based on known transition probabilities
            # compute whittle index for each arm

            indices = [(i,solve_gamma_2(env.transitions[i],discount,state[i])) for i in range(N)]
            indices = sorted(indices,key=lambda k: k[1],reverse=True)[:budget]
            indices = [i[0] for i in indices]
            action = np.zeros(N, dtype=np.int8)
            action[indices] = 1
            _, reward, done, _ = env.step(action)
            if done and t+1 < T: env.reset()
            all_reward[epoch, t] = reward

    return all_reward

def optimal_policy(env, n_episodes, n_epochs, discount,use_match_reward=False,lamb=0):
    """ optimal policy based on true transition probabilities """
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
                    state_WI[i] = arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,use_match_reward=use_match_reward)
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

def optimal_match_slow_policy(env, n_episodes, n_epochs, discount,lamb=0):
    """ optimal policy based on true transition probabilities
    
    We run Q-iteration on the full combinations of states 
    Hence this runs quite slowly, but provides a good upper bound
    For matching performance

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

        Q_vals = arm_value_iteration_match(true_transitions,discount,budget,match_prob,lamb=lamb)

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




