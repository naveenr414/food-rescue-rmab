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
from rmab.mcts_policies import mcts_policy, mcts_mcts_policy, mcts_whittle_policy
from rmab.fr_dynamics import get_all_transitions
from rmab.utils import get_save_path, delete_duplicate_results

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 42
    n_arms      = 2
    volunteers_per_arm = 16
    budget      = 3
    discount    = 0.9
    alpha       = 3 
    n_episodes  = 30
    episode_len = 20
    n_epochs    = 10
    save_with_date = False 
    TIME_PER_RUN = 0.01 * 1000
    save_name = 'combined_{}_{}_{}'.format(n_arms,volunteers_per_arm,seed)
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
    parser.add_argument('--time_per_run',      '-t', help='time per MCTS run', type=float, default=.01*1000)
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
    TIME_PER_RUN = args.time_per_run


# -

n_states = 2
n_actions = 2

all_population_size = 100 # number of random arms to generate
all_transitions = get_all_transitions(all_population_size)

random.seed(seed)
np.random.seed(seed)

all_features = np.arange(all_population_size)
match_probabilities = [random.random() for i in range(all_population_size * volunteers_per_arm)]

np.random.seed(seed)
random.seed(seed)
simulator = RMABSimulator(all_population_size, all_features, all_transitions,
            n_arms, volunteers_per_arm, episode_len, n_epochs, n_episodes, budget, discount,number_states=n_states, reward_style='match',match_probability_list=match_probabilities,TIME_PER_RUN=TIME_PER_RUN)

lamb = 1/(n_arms*volunteers_per_arm)

# ## Index Policies

if is_jupyter:
    policy = greedy_policy
    greedy_reward, greedy_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_greedy = simulator.time_taken
    print(np.mean(greedy_reward) + lamb*n_arms*volunteers_per_arm*greedy_active_rate)

if is_jupyter:
    policy = random_policy
    random_reward, random_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_random = simulator.time_taken
    print(np.mean(random_reward) + random_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = whittle_activity_policy
    whittle_reward, whittle_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_whittle = simulator.time_taken    
    print(np.mean(whittle_reward) + whittle_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = whittle_policy
    whittle_reward, whittle_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_whittle = simulator.time_taken    
    print(np.mean(whittle_reward) + whittle_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = whittle_whittle_policy
    whittle_whittle_reward, whittle_whittle_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_whittle = simulator.time_taken    
    print(np.mean(whittle_whittle_reward) + whittle_whittle_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = greedy_one_step_policy
    greedy_one_step_reward, greedy_one_step_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_greedy_one_step = simulator.time_taken
    print(np.mean(greedy_one_step_reward) + greedy_one_step_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = shapley_whittle_policy 
    whittle_shapley_reward, whittle_shapley_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_whittle_shapley = simulator.time_taken
    print(np.mean(whittle_shapley_reward) + whittle_shapley_active_rate*lamb*n_arms*volunteers_per_arm)
    

if is_jupyter:
    policy = whittle_greedy_policy 
    whittle_greedy_reward, whittle_greedy_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_whitte_greedy = simulator.time_taken
    print(np.mean(whittle_greedy_reward) + whittle_greedy_active_rate*lamb*n_arms*volunteers_per_arm)

# ## MCTS Policies

if is_jupyter:
    policy = mcts_policy 
    mcts_reward, mcts_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_mcts = simulator.time_taken
    print(np.mean(mcts_reward) + mcts_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = mcts_mcts_policy
    mcts_mcts_reward, mcts_mcts_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_mcts_mcts = simulator.time_taken
    print(np.mean(mcts_mcts_reward) + mcts_mcts_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter:
    policy = mcts_whittle_policy
    mcts_whittle_reward, mcts_whittle_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
    time_mcts_whittle = simulator.time_taken
    print(np.mean(mcts_whittle_reward) + mcts_whittle_active_rate*lamb*n_arms*volunteers_per_arm)

if is_jupyter and "time" in save_name:
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
            'lamb': lamb,
            'time_per_run': TIME_PER_RUN} 

    results['greedy_time'] = time_greedy
    results['greedy_score'] = np.mean(greedy_reward) + lamb*n_arms*volunteers_per_arm*greedy_active_rate
    results['whittle_time'] = time_whittle 
    results['whittle_score'] = np.mean(whittle_reward) + lamb*n_arms*volunteers_per_arm*whittle_active_rate
    results['greedy_one_step_time'] = time_greedy_one_step
    results['greedy_one_step_score'] = np.mean(greedy_one_step_reward) + lamb*n_arms*volunteers_per_arm*greedy_one_step_active_rate
    results['shapley_whittle_time'] = time_whittle_shapley
    results['shapley_whittle_score'] = np.mean(whittle_shapley_reward) + lamb*n_arms*volunteers_per_arm*whittle_shapley_active_rate
    results['whittle_greedy_time'] = time_whitte_greedy
    results['whittle_greedy_score'] = np.mean(whittle_greedy_reward) + lamb*n_arms*volunteers_per_arm*whittle_greedy_active_rate
    results['mcts_time'] = time_mcts 
    results['mcts_score'] = np.mean(mcts_reward) + lamb*n_arms*volunteers_per_arm*mcts_active_rate
    results['mcts_mcts_time'] = time_mcts_mcts 
    results['mcts_mcts_score'] = np.mean(mcts_mcts_reward) + lamb*n_arms*volunteers_per_arm*mcts_mcts_active_rate
    results['mcts_whittle_time'] = time_mcts_whittle
    results['mcts_whittle_score'] = np.mean(mcts_whittle_reward) + lamb*n_arms*volunteers_per_arm*mcts_whittle_active_rate
    save_path = get_save_path('combined_large',save_name,seed,use_date=save_with_date)
    json.dump(results,open('../results/'+save_path,'w'))

# ## Optimal Policy

if is_jupyter and n_arms*volunteers_per_arm <= 6:
    policy = q_iteration_policy
    per_epoch_function = q_iteration_epoch
    q_reward, q_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb,per_epoch_function=per_epoch_function)
    print(np.mean(q_reward) + q_active_rate*lamb*n_arms*volunteers_per_arm)

# ## Actual Experiments

if "two_step" in save_name:
    lamb_list = [1,16,64]
else:
    lamb_list = [0,0.25,0.5,1,2,4,8,16,32,64] 
lamb_list = [i/(n_arms*volunteers_per_arm) for i in lamb_list]

if "combined" in save_name:
    policies = [random_policy,greedy_policy,greedy_one_step_policy,whittle_policy,whittle_whittle_policy,whittle_activity_policy,shapley_whittle_policy,whittle_greedy_policy]
    policy_names = ["random","greedy","greedy_one_step","whittle","whittle_whittle","whittle_activity","shapley_whittle","whittle_greedy"]
elif "mcts" in save_name:
    policies = [mcts_policy,mcts_mcts_policy]
    policy_names = ["mcts","mcts_mcts"]
elif "two_step" in save_name:
    policies = [whittle_greedy_policy,mcts_policy,mcts_mcts_policy,mcts_whittle_policy]
    policy_names = ["whittle_greedy","mcts","mcts_mcts","mcts_whittle"]

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
        'lambda_list': lamb_list,
        'time_per_run': TIME_PER_RUN} 

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


