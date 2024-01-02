# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: food
#     language: python
#     name: python3
# ---

# # Large Combined Oracle Bandits

# Analyze the performance of various algorithms to solve the joint matching + activity task, when the number of volunteers is large and structured

# %load_ext autoreload
# %autoreload 2

import numpy as np
import random 
import matplotlib.pyplot as plt
import json 
import argparse 
import sys
import time 

from rmab.simulator import RMABSimulator
from rmab.omniscient_policies import *
from rmab.mcts_policies import mcts_policy, mcts_max_policy, mcts_greedy_policy, mcts_mcts_policy
from rmab.fr_dynamics import get_all_transitions
from rmab.utils import get_save_path, delete_duplicate_results

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 42
    n_arms      = 2
    volunteers_per_arm = 2
    budget      = 3
    discount    = 0.9
    alpha       = 3 
    n_episodes  = 30
    episode_len = 20 
    n_epochs    = 10
    save_name = 'mcts_{}_{}_{}'.format(n_arms,volunteers_per_arm,seed)
    save_with_date = False 
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=3)
    parser.add_argument('--volunteers_per_arm',         '-V', help='volunteers per arm', type=int, default=2)
    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=20)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=30)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=3)
    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=10)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.9)
    parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=42)
    parser.add_argument('--save_name',      '-n', help='save name', type=str, default='combined_lamb')
    parser.add_argument('--use_date', action='store_true')

    args = parser.parse_args()

    n_arms      = args.n_arms
    volunteers_per_arm = args.volunteers_per_arm
    budget      = args.budget
    discount    = args.discount
    alpha       = args.alpha 
    seed        = args.seed
    n_episodes  = args.n_episodes
    episode_len = args.episode_len
    n_epochs    = args.n_epochs
    save_name   = args.save_name 
    save_with_date = args.use_date 


# -

n_states = 2
n_actions = 2

all_population_size = 100 # number of random arms to generate
all_transitions = get_all_transitions(all_population_size)

all_features = np.arange(all_population_size)
match_probabilities = [random.random() for i in range(all_population_size * volunteers_per_arm)]

np.random.seed(seed)
random.seed(seed)
simulator = RMABSimulator(all_population_size, all_features, all_transitions,
            n_arms, volunteers_per_arm, episode_len, n_epochs, n_episodes, budget, discount,number_states=n_states, reward_style='match',match_probability_list=match_probabilities)

lamb = 0 # 1/(n_arms*volunteers_per_arm)

# ## Index Policies

if is_jupyter:
    policy = greedy_policy
    greedy_reward, greedy_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(greedy_reward) + lamb*n_arms*volunteers_per_arm*greedy_active_rate)

if is_jupyter:
    policy = random_policy
    random_reward, random_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(random_reward) + random_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = whittle_policy
    whittle_reward, whittle_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(whittle_reward) + whittle_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = greedy_one_step_policy
    greedy_one_step_reward, greedy_one_step_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(greedy_one_step_reward) + greedy_one_step_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = shapley_whittle_policy 
    whittle_shapley_reward, whittle_shapley_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(whittle_shapley_reward) + whittle_shapley_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = whittle_greedy_policy 
    whittle_greedy_reward, whittle_greedy_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(whittle_greedy_reward) + whittle_greedy_active_rate*lamb*n_arms*volunteers_per_arm)

# ## MCTS Policies

if is_jupyter:
    policy = mcts_policy 
    mcts_reward, mcts_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(mcts_reward) + mcts_active_rate*lamb*n_arms*volunteers_per_arm)


if is_jupyter:
    policy = mcts_max_policy
    mcts_max_reward, mcts_max_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(mcts_max_reward) + mcts_max_active_rate*lamb*n_arms*volunteers_per_arm)


if is_jupyter:
    policy = mcts_greedy_policy
    mcts_greedy_reward, mcts_greedy_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(mcts_greedy_reward) + mcts_greedy_active_rate*lamb*n_arms*volunteers_per_arm)


if is_jupyter:
    policy = mcts_mcts_policy
    mcts_mcts_reward, mcts_mcts_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(mcts_mcts_reward) + mcts_mcts_active_rate*lamb*n_arms*volunteers_per_arm)


# ## Optimal Policy

if is_jupyter:
    policy = q_iteration_policy
    per_epoch_function = q_iteration_epoch
    q_reward, q_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb,per_epoch_function=per_epoch_function)
    print(np.mean(q_reward) + q_active_rate*lamb*n_arms*volunteers_per_arm)

# ## Index Policies, Large N

if is_jupyter:
    n_arms = 3 
    volunteers_per_arm = 10
    match_probabilities = [random.random() for i in range(all_population_size * volunteers_per_arm)]

    np.random.seed(seed)
    random.seed(seed)
    simulator = RMABSimulator(all_population_size, all_features, all_transitions,
                n_arms, volunteers_per_arm, episode_len, n_epochs, n_episodes, budget, discount,number_states=n_states, reward_style='match',match_probability_list=match_probabilities)

lamb = 1/(n_arms*volunteers_per_arm)

if is_jupyter:
    start = time.time()
    policy = random_policy
    random_reward, random_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(random_reward) + random_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

if is_jupyter:
    start = time.time()
    policy = whittle_policy
    whittle_reward, whittle_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(whittle_reward) + whittle_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

if is_jupyter:
    start = time.time()
    policy = greedy_one_step_policy
    greedy_one_step_reward, greedy_one_step_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(greedy_one_step_reward) + greedy_one_step_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

if is_jupyter:
    start = time.time()
    policy = shapley_whittle_policy 
    whittle_shapley_reward, whittle_shapley_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(whittle_shapley_reward) + whittle_shapley_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

if is_jupyter:
    start = time.time()
    policy = whittle_greedy_policy 
    whittle_greedy_reward, whittle_greedy_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(whittle_greedy_reward) + whittle_greedy_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

if is_jupyter:
    n_arms = 10
    volunteers_per_arm = 3
    match_probabilities = [random.random() for i in range(all_population_size * volunteers_per_arm)]

    np.random.seed(seed)
    random.seed(seed)
    simulator = RMABSimulator(all_population_size, all_features, all_transitions,
                n_arms, volunteers_per_arm, episode_len, n_epochs, n_episodes, budget, discount,number_states=n_states, reward_style='match',match_probability_list=match_probabilities)

lamb = 1/(n_arms*volunteers_per_arm)

if is_jupyter:
    start = time.time()
    policy = greedy_policy
    greedy_reward, greedy_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(greedy_reward) + greedy_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

if is_jupyter:
    start = time.time()
    policy = whittle_policy
    whittle_reward, whittle_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(whittle_reward) + whittle_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

if is_jupyter:
    start = time.time()
    policy = greedy_one_step_policy
    greedy_one_step_reward, greedy_one_step_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(greedy_one_step_reward) + greedy_one_step_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

if is_jupyter:
    start = time.time()
    policy = whittle_activity_policy   
    whittle_activity_reward, whittle_activity_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(whittle_activity_reward) + whittle_activity_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

if is_jupyter:
    start = time.time()
    policy = shapley_whittle_policy 
    whittle_shapley_reward, whittle_shapley_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(whittle_shapley_reward) + whittle_shapley_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

if is_jupyter:
    start = time.time()
    policy = whittle_greedy_policy 
    whittle_greedy_reward, whittle_greedy_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    print(np.mean(whittle_greedy_reward) + whittle_greedy_active_rate*lamb*n_arms*volunteers_per_arm)
    print("Took {} time".format(time.time()-start))

# ## Actual Experiments

lamb_list = [0,0.25,0.5,1,2,4,8,16,32,64] 
lamb_list = [i/(n_arms*volunteers_per_arm) for i in lamb_list]

if "combined" in save_name:
    policies = [random_policy,greedy_policy,greedy_one_step_policy,whittle_policy,whittle_activity_policy,shapley_whittle_policy,whittle_greedy_policy]
    policy_names = ["random","greedy","greedy_one_step","whittle","whittle_activity","shapley_whittle","whittle_greedy"]
elif "mcts" in save_name:
    policies = [mcts_policy,mcts_mcts_policy]
    policy_names = ["mcts","mcts_mcts"]

results = {}
results['parameters'] = {'seed'      : seed,
        'n_arms'    : n_arms,
        'volunteers_per_arm': volunteers_per_arm, 
        'budget'    : budget,
        'discount'  : discount, 
        'alpha'     : alpha, 
        'n_episodes': n_episodes, 
        'episode_len': episode_len, 
        'n_epochs'  : n_epochs, 
        'lambda_list': lamb_list,} 

if (n_arms * volunteers_per_arm) <= 6:
    print("Running optimal")
    policy = q_iteration_policy
    per_epoch_function = q_iteration_epoch
    
    match_reward_list = []
    active_rate_list = []

    for lamb in lamb_list:
        reward, active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb,per_epoch_function=per_epoch_function)
        match_reward_list.append(np.mean(reward))
        active_rate_list.append(active_rate)

    results['optimal_match'] = match_reward_list 
    results['optimal_active'] = active_rate_list     

for policy,name in zip(policies,policy_names):
    match_reward_list = []
    active_rate_list = []

    print("On policy {}".format(name))

    for lamb in lamb_list:
        reward, active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
        match_reward_list.append(np.mean(reward))
        active_rate_list.append(active_rate)

    results['{}_match'.format(name)] = match_reward_list 
    results['{}_active'.format(name)] = active_rate_list 

save_path = get_save_path('combined_large',save_name,seed,use_date=save_with_date)

delete_duplicate_results('combined_large',save_name,results)

json.dump(results,open('../results/'+save_path,'w'))


