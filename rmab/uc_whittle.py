""" UC Whittle """

import sys
import warnings
import random
import numpy as np
import heapq  # priority queue
from scipy.stats import beta 

import gurobipy as gp
from gurobipy import GRB

from rmab.simulator import RMABSimulator, random_valid_transition
from rmab.compute_whittle import arm_compute_whittle
from rmab.utils import Memoizer, get_valid_lcb_ucb, get_ucb_conf, get_stationary_distribution

def reward(s):
    """ reward for a state is just its own value (reward is 1 for s=1, 0 for s=0) """
    return s


def UCWhittle(env, n_episodes, n_epochs, discount, alpha, method='QP', VERBOSE=False):
    """
    discount = discount factor
    alpha = for confidence radius """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    print(f'solving UCWhittle using method: {method}')

    all_reward = np.zeros((n_epochs, T))

    memoizer = Memoizer(method)

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()

        n_pulls  = np.zeros((N, n_states, n_actions))  # number of pulls
        cum_prob = np.zeros((N, n_states, n_actions))  # transition probability estimates for going to ENGAGING state

        print('first state', env.observe())

        for t in range(0,T):
            state = env.observe()

            est_p, conf_p = get_ucb_conf(cum_prob, n_pulls, t, alpha, env.episode_count)

            # TODO: use complements for transition probabilities? w.r.t. (action, s' = {0, 1})
            # idea: we want to be optimistic about next state being ENGAGING (and then the complement for NE)

            if method == 'QP':         # quadratic constraint program (QCP)
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer)
            elif method == 'QP-min':         # quadratic constraint program (QCP) with minimizing lambda
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer, approach='min')
            elif method == 'extreme':  # extreme points
                action, state_WI = extreme_points_step(est_p, conf_p, state, discount, budget, memoizer)
            elif method == 'UCB':      # only UCB estimates
                action, state_WI = UCB_step(est_p, conf_p, state, discount, budget, memoizer)

            # execute chosen policy; observe reward and next state
            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            # update estimates
            for i in range(N):
                # TODO: This is the bad transition way; this is only
                # to prove that we fixed the issue

                for s in range(n_states): # pick states with B largest WI
                    a = action[s]

                    n_pulls[i, s, a] += 1
                    if next_state[i] == 1:
                        cum_prob[i, s, a] += 1

            # print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | Q_active ', np.round(Q_active[:, state], 3), ' | Q_passive ', np.round(Q_passive, 3), ' | WI ', np.round(state_WI, 3))
            if t % 100 == 0:
                print('---------------------------------------------------')
                print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | WI ', np.round(state_WI, 3))

            all_reward[epoch, t] = reward

    return all_reward

def UCWhittleFixed(env, n_episodes, n_epochs, discount, alpha, method='QP', VERBOSE=False,norm_confidence=False,delta=1e-4):
    """
    discount = discount factor
    alpha = for confidence radius """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    print(f'solving UCWhittle using method: {method}')

    all_reward = np.zeros((n_epochs, T))

    memoizer = Memoizer(method)

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()

        n_pulls  = np.zeros((N, n_states, n_actions))  # number of pulls
        cum_prob = np.zeros((N, n_states, n_actions))  # transition probability estimates for going to ENGAGING state

        print('first state', env.observe())

        for t in range(0,T):
            state = env.observe()

            est_p, conf_p = get_ucb_conf(cum_prob, n_pulls, t, alpha, env.episode_count,norm_confidence=norm_confidence,delta=delta)

            # TODO: use complements for transition probabilities? w.r.t. (action, s' = {0, 1})
            # idea: we want to be optimistic about next state being ENGAGING (and then the complement for NE)

            if method == 'QP':         # quadratic constraint program (QCP)
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer)
            elif method == 'QP-min':         # quadratic constraint program (QCP) with minimizing lambda
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer, approach='min')
            elif method == 'extreme':  # extreme points
                action, state_WI = extreme_points_step(est_p, conf_p, state, discount, budget, memoizer)
            elif method == 'UCB':      # only UCB estimates
                action, state_WI = UCB_step(est_p, conf_p, state, discount, budget, memoizer)

            # execute chosen policy; observe reward and next state
            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            # update estimates
            for i in range(N):
                s = state[i] 
                a = action[i] 

                n_pulls[i, s, a] += 1
                if next_state[i] == 1:
                    cum_prob[i, s, a] += 1

            # print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | Q_active ', np.round(Q_active[:, state], 3), ' | Q_passive ', np.round(Q_passive, 3), ' | WI ', np.round(state_WI, 3))
            if t % 100 == 0:
                print('---------------------------------------------------')
                print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | WI ', np.round(state_WI, 3))

            all_reward[epoch, t] = reward

    return all_reward

def UCWhittlePerfect(env, n_episodes, n_epochs, discount, alpha, method='QP', VERBOSE=False,norm_confidence=False,delta=1e-4):
    """
    discount = discount factor
    alpha = for confidence radius """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    print(f'solving UCWhittle using method: {method}')

    all_reward = np.zeros((n_epochs, T))

    memoizer = Memoizer(method)

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()

        n_pulls  = np.zeros((N, n_states, n_actions))  # number of pulls
        cum_prob = np.zeros((N, n_states, n_actions))  # transition probability estimates for going to ENGAGING state

        print('first state', env.observe())

        for t in range(0,T):
            state = env.observe()

            n_pulls_at_least_1 = np.copy(n_pulls)
            n_pulls_at_least_1[n_pulls == 0] = 1
            est_p               = (cum_prob) / (n_pulls_at_least_1)
            est_p[n_pulls == 0] = 1 / n_states  # where division by 0
            conf_p = np.abs(est_p-env.transitions[:,:,:,1])

            # TODO: use complements for transition probabilities? w.r.t. (action, s' = {0, 1})
            # idea: we want to be optimistic about next state being ENGAGING (and then the complement for NE)

            if method == 'QP':         # quadratic constraint program (QCP)
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer)
            elif method == 'QP-min':         # quadratic constraint program (QCP) with minimizing lambda
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer, approach='min')
            elif method == 'extreme':  # extreme points
                action, state_WI = extreme_points_step(est_p, conf_p, state, discount, budget, memoizer)
            elif method == 'UCB':      # only UCB estimates
                action, state_WI = UCB_step(est_p, conf_p, state, discount, budget, memoizer)

            # execute chosen policy; observe reward and next state
            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            # update estimates
            for i in range(N):
                s = state[i] 
                a = action[i] 

                n_pulls[i, s, a] += 1
                if next_state[i] == 1:
                    cum_prob[i, s, a] += 1

            # print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | Q_active ', np.round(Q_active[:, state], 3), ' | Q_passive ', np.round(Q_passive, 3), ' | WI ', np.round(state_WI, 3))
            if t % 100 == 0:
                print('---------------------------------------------------')
                print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | WI ', np.round(state_WI, 3))

            all_reward[epoch, t] = reward

    return all_reward


def UCWhittleMatch(env, n_episodes, n_epochs, discount, alpha, method='QP', VERBOSE=False,delta=1e-4):
    """
    discount = discount factor
    alpha = for confidence radius """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    print(f'solving UCWhittle using method: {method}')

    all_reward = np.zeros((n_epochs, T))

    memoizer = Memoizer(method)

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()

        n_pulls  = np.zeros((N, n_states, n_actions))  # number of pulls
        cum_prob = np.zeros((N, n_states, n_actions))  # transition probability estimates for going to ENGAGING state

        print('first state', env.observe())


        for t in range(0,T):
            state = env.observe()

            est_p, conf_p = get_ucb_conf(cum_prob, n_pulls, t, alpha, env.episode_count,norm_confidence=True,delta=delta)

            # TODO: use complements for transition probabilities? w.r.t. (action, s' = {0, 1})
            # idea: we want to be optimistic about next state being ENGAGING (and then the complement for NE)

            if method == 'QP':         # quadratic constraint program (QCP)
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer,use_match=True)
            elif method == 'QP-min':         # quadratic constraint program (QCP) with minimizing lambda
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer, approach='min',use_match=True)
            elif method == 'extreme':  # extreme points
                action, state_WI = extreme_points_step(est_p, conf_p, state, discount, budget, memoizer,use_match=True)
            elif method == 'UCB':      # only UCB estimates
                action, state_WI = UCB_step(est_p, conf_p, state, discount, budget, memoizer,use_match=True)

            # execute chosen policy; observe reward and next state
            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            # update estimates
            for i in range(N):
                s = state[i] 
                a = action[i] 

                n_pulls[i, s, a] += 1
                if next_state[i] == 1:
                    cum_prob[i, s, a] += 1

            # print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | Q_active ', np.round(Q_active[:, state], 3), ' | Q_passive ', np.round(Q_passive, 3), ' | WI ', np.round(state_WI, 3))
            if t % 100 == 0:
                print('---------------------------------------------------')
                print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | WI ', np.round(state_WI, 3))

            all_reward[epoch, t] = reward

    return all_reward

def UCWhittlePerfectMatch(env, n_episodes, n_epochs, discount, alpha, method='QP', VERBOSE=False,delta=1e-4):
    """
    discount = discount factor
    alpha = for confidence radius """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    print(f'solving UCWhittle using method: {method}')

    all_reward = np.zeros((n_epochs, T))

    memoizer = Memoizer(method)

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()

        n_pulls  = np.zeros((N, n_states, n_actions))  # number of pulls
        cum_prob = np.zeros((N, n_states, n_actions))  # transition probability estimates for going to ENGAGING state

        print('first state', env.observe())


        for t in range(0,T):
            state = env.observe()

            n_pulls_at_least_1 = np.copy(n_pulls)
            n_pulls_at_least_1[n_pulls == 0] = 1
            est_p               = (cum_prob) / (n_pulls_at_least_1)
            est_p[n_pulls == 0] = 1 / n_states  # where division by 0
            conf_p = np.abs(est_p-env.transitions[:,:,:,1])

            # TODO: use complements for transition probabilities? w.r.t. (action, s' = {0, 1})
            # idea: we want to be optimistic about next state being ENGAGING (and then the complement for NE)

            if method == 'QP':         # quadratic constraint program (QCP)
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer,use_match=True)
            elif method == 'QP-min':         # quadratic constraint program (QCP) with minimizing lambda
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer, approach='min',use_match=True)
            elif method == 'extreme':  # extreme points
                action, state_WI = extreme_points_step(est_p, conf_p, state, discount, budget, memoizer,use_match=True)
            elif method == 'UCB':      # only UCB estimates
                action, state_WI = UCB_step(est_p, conf_p, state, discount, budget, memoizer,use_match=True)

            # execute chosen policy; observe reward and next state
            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            # update estimates
            for i in range(N):
                s = state[i] 
                a = action[i] 

                n_pulls[i, s, a] += 1
                if next_state[i] == 1:
                    cum_prob[i, s, a] += 1

            # print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | Q_active ', np.round(Q_active[:, state], 3), ' | Q_passive ', np.round(Q_passive, 3), ' | WI ', np.round(state_WI, 3))
            if t % 100 == 0:
                print('---------------------------------------------------')
                print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | WI ', np.round(state_WI, 3))

            all_reward[epoch, t] = reward

    return all_reward


def NStepMatch(env, n_episodes, n_epochs, discount, alpha,n_step, method='QP', VERBOSE=False):
    """Function to determine optimal matches using transition probabilities 
        + Stationary Distribution, using Normal Confidence Intervals
    Arguments:
        env: RMAB environment
        n_episodes: Integer, number of episodes
        n_epochs: Integer, number of epochs
        discount: Float, \gamma, how much to discount
        alpha: Used to compute the confidence interval
        
    Returns: Rewards for each epoch"""
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    print(f'solving UCWhittle using method: {method}')

    all_reward = np.zeros((n_epochs, T))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()

        n_pulls  = np.zeros((N, n_states, n_actions))  # number of pulls
        cum_prob = np.zeros((N, n_states, n_actions))  # transition probability estimates for going to ENGAGING state

        print('first state', env.observe())

        for t in range(0, T):
            state = env.observe()

            est_p, conf_p = get_ucb_conf(cum_prob, n_pulls, t, alpha, env.episode_count,norm_confidence=True)

            # est_p is for statexaction
            full_transition = np.stack((1-est_p,est_p),axis=-1)
            full_conf = np.stack((conf_p,conf_p),axis=-1)

            max_p = full_transition
            max_p[:,:,1,1] += full_conf[:,:,1,1] 
            
            for i in range(len(max_p)):
                for s in range(len(max_p[i])):
                    for a in range(len(max_p[i][s])):
                        max_p[i][s][a]/=np.sum(max_p[i][s][a])

            if n_step == 0 or n_step == 1:
                power_transitions = [t[:,1,:] for t in max_p]
            # Look ahead more than 1 step 
            elif n_step > 1:
                power_transitions = [np.linalg.matrix_power(t[:,1,:],n_step) for t in max_p] 
            # Look ahead infinite steps (steady state)
            elif n_step == -1:
                stationary_distribution = [get_stationary_distribution(t[:,1,:]) for t in max_p]
                power_transitions = [[r,r] for r in stationary_distribution]

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

            # execute chosen policy; observe reward and next state
            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            # update estimates
            for i in range(N):
                s = state[i] 
                a = action[i] 

                n_pulls[i, s, a] += 1
                if next_state[i] == 1:
                    cum_prob[i, s, a] += 1

            # print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | Q_active ', np.round(Q_active[:, state], 3), ' | Q_passive ', np.round(Q_passive, 3), ' | WI ', np.round(state_WI, 3))
            if t % 100 == 0:
                print('---------------------------------------------------')
                print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward)

            all_reward[epoch, t] = reward

    return all_reward

def NormPlusMatch(env, n_episodes, n_epochs, discount, alpha, method='QP', VERBOSE=False,lamb=0.5):
    """
    discount = discount factor
    alpha = for confidence radius """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    print(f'solving UCWhittle using method: {method}')

    all_reward = np.zeros((n_epochs, T + 1))

    memoizer = Memoizer(method)

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()

        n_pulls  = np.zeros((N, n_states, n_actions))  # number of pulls
        cum_prob = np.zeros((N, n_states, n_actions))  # transition probability estimates for going to ENGAGING state

        print('first state', env.observe())
        all_reward[epoch, 0] = env.get_reward()

        for t in range(0, T):
            state = env.observe()

            est_p, conf_p = get_ucb_conf(cum_prob, n_pulls, t, alpha, env.episode_count,norm_confidence=True)

            # TODO: use complements for transition probabilities? w.r.t. (action, s' = {0, 1})
            # idea: we want to be optimistic about next state being ENGAGING (and then the complement for NE)

            if method == 'QP':         # quadratic constraint program (QCP)
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer)
            elif method == 'QP-min':         # quadratic constraint program (QCP) with minimizing lambda
                action, state_WI = QP_step(est_p, conf_p, state, discount, budget, memoizer, approach='min')
            elif method == 'extreme':  # extreme points
                action, state_WI = extreme_points_step(est_p, conf_p, state, discount, budget, memoizer)
            elif method == 'UCB':      # only UCB estimates
                action, state_WI = UCB_step(est_p, conf_p, state, discount, budget, memoizer)

            full_transition = np.stack((1-est_p,est_p),axis=-1)
            full_conf = np.stack((conf_p,conf_p),axis=-1)

            max_p = full_transition
            max_p[:,:,1,1] += full_conf[:,:,1,1] 
            
            for i in range(len(max_p)):
                for s in range(len(max_p[i])):
                    for a in range(len(max_p[i][s])):
                        max_p[i][s][a]/=np.sum(max_p[i][s][a])

            stationary_distros = [get_stationary_distribution(i[:,1,:]) for i in max_p]
            chance_in_1 = [(prob[1],i) for i,prob in enumerate(stationary_distros)]

            matching_values = [i[0] for i in chance_in_1] 
            whittle_values = state_WI 

            matching_values = np.array(matching_values)
            whittle_values = np.array(whittle_values )

            matching_values = np.clip(matching_values,0,None)
            whittle_values = np.clip(whittle_values,0,None)
            matching_values /= np.max(matching_values)
            whittle_values /= np.max(whittle_values)

            matching_values *= lamb 

            values_combined = matching_values + whittle_values 
            indices_combined = np.argsort(values_combined)[-budget:]

            action_match = np.zeros(N,dtype=np.int8)
            action_match[np.array(indices_combined)] = 1

            # execute chosen policy; observe reward and next state
            next_state, reward, done, _ = env.step(action_match)

            if done and t+1 < T: env.reset()

            # update estimates
            for i in range(N):
                s = state[i] 
                a = action[i] 

                n_pulls[i, s, a] += 1
                if next_state[i] == 1:
                    cum_prob[i, s, a] += 1

            # print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | Q_active ', np.round(Q_active[:, state], 3), ' | Q_passive ', np.round(Q_passive, 3), ' | WI ', np.round(state_WI, 3))
            if t % 100 == 0:
                print('---------------------------------------------------')
                print(epoch, t, ' | a ', action, ' | s\' ', next_state, ' | r ', reward, '   | WI ', np.round(state_WI, 3))

            all_reward[epoch, t] = reward

    return all_reward





#####################################################################
# specific implementations for choosing action within UCWhittle
#####################################################################

def extreme_points_step(est_p, conf_p, state, discount, budget, memoizer):
    """ step of UCWhittle using only extreme points """

    # solve MIP giving values to each of the extreme points... simultaneously
    N, n_states, n_actions = est_p.shape

    est_transitions = np.zeros((N, n_states, n_actions))
    est_transitions[:, :, 1] = est_p[:, :, 1] + conf_p[:, :, 1]  # for active action, take UCB
    est_transitions[:, :, 0] = est_p[:, :, 0] - conf_p[:, :, 0]  # for passive action, take LCB

    est_transitions[est_transitions > 1] = 1  # keep within valid range
    est_transitions[est_transitions < 0] = 0  # keep within valid range

    # if VERBOSE: print('est transitions', np.round(est_transitions.flatten(), 2))

    assert np.all(est_transitions <= 1), print(est_transitions)

    # compute whittle index for each arm
    state_WI = np.zeros(N)
    top_WI = []
    min_chosen_subsidy = 0
    for i in range(N):
        arm_transitions = est_transitions[i, :, :]
        # memoize to speed up
        check_set_val = memoizer.check_set(arm_transitions, state[i])
        if check_set_val != -1:
            state_WI[i] = check_set_val
        else:
            state_WI[i] = arm_compute_whittle(arm_transitions, state[i], discount, subsidy_break=min_chosen_subsidy)
            memoizer.add_set(arm_transitions, state[i], state_WI[i])


        if len(top_WI) < budget:
            heapq.heappush(top_WI, (state_WI[i], i))
        else:
            # add state_WI to heap
            heapq.heappushpop(top_WI, (state_WI[i], i))
            min_chosen_subsidy = top_WI[0][0]  # smallest-valued item


    # pull K highest indices
    sorted_WI = np.argsort(state_WI)[::-1]
    # print('state_WI', np.round(state_WI, 2))

    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, state_WI



def UCB_step(est_p, conf_p, state, discount, budget, memoizer,use_match=False):
    """ a single step for UCWhittle: determine action
    given current TP estimates
    using only UCB estimates (upper bound) """

    N, n_states, n_actions = est_p.shape

    est_transitions = est_p + conf_p
    est_transitions[est_transitions > 1] = 1  # keep within valid range

    # assert np.all(est_transitions <= 1), print(est_transitions)

    # print('\n\n#######################')

    # compute whittle index for each arm
    state_WI = np.zeros(N)
    top_WI = []
    min_chosen_subsidy = 0
    for i in range(N):
        arm_transitions = est_transitions[i, :, :]
        # memoize to speed up
        check_set_val = memoizer.check_set(arm_transitions, state[i])
        if check_set_val != -1:
            state_WI[i] = check_set_val
            # print(f'  >> skipping! arm {i} WI {state_WI[i]:.3f} lcb = {arm_p_lcb.flatten().round(3)}, ucb = {arm_p_ucb.flatten().round(3)}')

        else:
            state_WI[i] = arm_compute_whittle(arm_transitions, state[i], discount, subsidy_break=min_chosen_subsidy,use_match_reward=use_match)
            memoizer.add_set(arm_transitions, state[i], state_WI[i])

        if len(top_WI) < budget:
            heapq.heappush(top_WI, (state_WI[i], i))
        else:
            # add state_WI to heap
            heapq.heappushpop(top_WI, (state_WI[i], i))
            min_chosen_subsidy = top_WI[0][0]  # smallest-valued item


    # pull K highest indices
    sorted_WI = np.argsort(state_WI)[::-1]
    # print('state_WI', np.round(state_WI, 2))

    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, state_WI



def QP_step(est_p, conf_p, state, discount, budget, memoizer, approach=None):
    """ step of UCWhittle using QP solver """

    N, n_states, n_actions = est_p.shape

    # compute UCB and LCB estimates of transition probabilities
    p_ucb = est_p + conf_p
    p_ucb[p_ucb > 1] = 1  # keep within valid range
    p_lcb = est_p - conf_p
    p_lcb[p_lcb < 0] = 0  # keep within valid range

    state_WI = np.zeros(N)
    top_WI = []
    min_chosen_subsidy = 0
    for i in range(N):
        arm_p_lcb, arm_p_ucb = get_valid_lcb_ucb(p_lcb[i, :, :], p_ucb[i, :, :])

        # memoize to speed up
        check_set_val = memoizer.check_set(arm_p_lcb, arm_p_ucb)
        if check_set_val != -1:
            state_WI[i] = check_set_val
            # print(f'  >> skipping! arm {i} WI {state_WI[i]:.3f} lcb = {arm_p_lcb.flatten().round(3)}, ucb = {arm_p_ucb.flatten().round(3)}')

        else:
            state_WI[i] = solve_QP_per_arm(arm_p_lcb, arm_p_ucb, state[i], discount, subsidy_break=min_chosen_subsidy, approach=approach)
            # print(f'  arm {i} WI: {state_WI[i]:.3f}, state {state[i]}, lcb = {arm_p_lcb.flatten().round(3)}, ucb = {arm_p_ucb.flatten().round(3)}')
            memoizer.add_set(arm_p_lcb, arm_p_ucb, state_WI[i])

        if len(top_WI) < budget:
            heapq.heappush(top_WI, (state_WI[i], i))
        else:
            # add state_WI to heap
            heapq.heappushpop(top_WI, (state_WI[i], i))
            min_chosen_subsidy = top_WI[0][0]  # smallest-valued item


    # get action corresponding to optimal subsidy
    # pull K highest indices
    sorted_WI = np.argsort(state_WI)[::-1]

    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, state_WI


def solve_QP_per_arm(p_lcb, p_ucb, s0, discount, subsidy_break, approach):
    """ solve QP to compute Whittle index for each arm independently
    s0: initial state of arm """

    # add padding to address numerical instability
    epsilon = 1e-4
    p_ucb = p_ucb + epsilon
    p_lcb = p_lcb - epsilon

    max_iterations = 100 # max number of simplex iterations

    n_states, n_actions = p_lcb.shape


    def early_termination(what, where):
        """ callback for gurobi early termination, to incorporate subsidy_break
        callback codes: https://www.gurobi.com/documentation/9.5/refman/cb_codes.html
        """

        if where == GRB.Callback.MIP:
            objbnd = what.cbGet(GRB.Callback.MIP_OBJBND)
            if objbnd < subsidy_break:
                # print(f'  gurobi terminate! {objbnd:.3f}')
                what.terminate()
        elif where == GRB.Callback.MIPSOL:
            objbnd = what.cbGet(GRB.Callback.MIPSOL_OBJBST)
            if objbnd < subsidy_break:
                # print(f'  gurobi terminate! {objbnd:.3f}')
                what.terminate()

    # set up Gurobi optimizer --------------------------------------------------
    model = gp.Model('UCWhittle_QP')
    model.setParam('OutputFlag', 0) # silence output
    model.setParam('NonConvex', 2)  # nonconvex constraints
    model.setParam('IterationLimit', max_iterations) # limit number of simplex iterations

    # define variables ---------------------------------------------------------
    # variables to estimate transition probability (within LCB and UCB)
    p        = [[model.addVar(vtype=GRB.CONTINUOUS, lb=p_lcb[s][a], ub=p_ucb[s][a], name=f'p_{s}{a}')
                for a in range(n_actions)] for s in range(n_states)]

    # variable to learn subsidy
    subsidy  = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=100, name='subsidy')  #GRB.INFINITY --> gives status 5 (unbounded)

    # variable for value functions
    value_sa = [[model.addVar(vtype=GRB.CONTINUOUS, name=f'v_{s}_{a}')
                for a in range(n_actions)] for s in range(n_states)]
    value_s  = [model.addVar(vtype=GRB.CONTINUOUS, name=f'v_{s}')
                for s in range(n_states)]


    # define objective ---------------------------------------------------------
    if approach == 'min': # try subsidy-minimizing QP
        model.setObjective(subsidy, GRB.MINIMIZE)
    else:
        model.setObjective(subsidy, GRB.MAXIMIZE)


    # define constraints -------------------------------------------------------
    # value function (s, a)
    model.addConstrs(((value_sa[s][a] == subsidy * (a == 0) + reward(s) + discount * (value_s[1] * p[s][a] +
                                                                                      value_s[0] * (1 - p[s][a])))
        for s in range(n_states) for a in range(n_actions)), 'value_sa')

    # value function (s)
    model.addConstrs(((value_s[s] == gp.max_([value_sa[s][0], value_sa[s][1]]))
                       # value_s[s] == gp.max_([value_sa[s][a] for a in range(n_actions)]))
        for s in range(n_states)), 'value_s')

    # add constraints that enforce probability validity
    model.addConstrs((p[s][1] >= p[s][0] for s in range(n_states)), 'good_to_act')  # always good to act
    model.addConstrs((p[1][a] >= p[0][a] for a in range(n_actions)), 'good_state')  # always good to be in good state

    # valid subsidy
    model.addConstr((value_sa[s0][0] <= value_sa[s0][1]), 'valid_subsidy')


    # optimize -----------------------------------------------------------------
    model.optimize(early_termination)
    

    if model.status != GRB.OPTIMAL:
        warnings.warn(f'Uh oh! Model is not optimal; status is {model.status}')

    if model.status == GRB.INTERRUPTED:
        return -1  # early termination due to subsidy break

    # get optimal subsidy
    try:
        opt_subsidy = subsidy.x
    except: # Error as err:
        print(f'can\'t get subsidy. model status = {model.status}, state {s0}, lcb = {p_lcb.flatten().round(3)}, ucb = {p_ucb.flatten().round(3)}')

        return subsidy.ub

    return opt_subsidy


if __name__ == '__main__':
    VERBOSE = False
    seed    = 42

    np.random.seed(seed)
    random.seed(seed)

    # initialize RMAB simulator
    all_population  = 2 #10#10 # 10000  # num beneficiaries
    cohort_size     = 2 #10 #200  # N
    episode_len     = 100 # horizon T
    budget          = 1 #20
    n_states        = 2
    n_actions       = 2
    all_features    = np.arange(all_population)  # (N, S, A, S')
    all_transitions = random_valid_transition(all_population, n_states, n_actions)

    # import pdb; pdb.set_trace()

    if VERBOSE: print(f'transitions ----------------\n{np.round(all_transitions, 2)}')


    simulator = RMABSimulator(all_population, all_features, all_transitions, cohort_size, episode_len, budget, number_states=n_states)

    N = cohort_size

    UCWhittle(simulator, discount=.9, alpha=.1, VERBOSE=VERBOSE)
