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

# # Combined Oracle Bandits

# Analyze the performance of various oracle bandits that solve the combined activity and matching task

# %load_ext autoreload
# %autoreload 2

import numpy as np
import random 
import matplotlib.pyplot as plt
import json 
import argparse 
import sys
from openrl.envs.common import make
from gymnasium.envs.registration import register

from rmab.simulator import RMABSimulator
from rmab.baselines import optimal_whittle,  optimal_q_iteration, optimal_whittle_sufficient, greedy_policy, random_policy, greedy_iterative_policy, mcts_policy
from rmab.fr_dynamics import get_all_transitions
from rmab.compute_whittle import arm_value_iteration_exponential
from rmab.utils import get_save_path, delete_duplicate_results, filter_pareto_optimal, is_pareto_optimal


is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 42
    n_arms      = 4
    budget      = 3 
    discount    = 0.9
    alpha       = 3 
    n_episodes  = 30
    episode_len = 20 
    n_epochs    = 10
    save_name = 'heterogenous_arms_{}'.format(n_arms)
    match_prob = 0.5
    save_with_date = False 
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=8)
    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=20)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=30)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=3)
    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=10)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.9)
    parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=42)
    parser.add_argument('--save_name',      '-n', help='save name', type=str, default='combined_lamb')
    parser.add_argument('--match_prob',      '-m', help='match probability', type=float, default=0.5)
    parser.add_argument('--use_date', action='store_true')

    args = parser.parse_args()

    n_arms      = args.n_arms
    budget      = args.budget
    discount    = args.discount
    alpha       = args.alpha 
    seed        = args.seed
    n_episodes  = args.n_episodes
    episode_len = args.episode_len
    n_epochs    = args.n_epochs
    save_name   = args.save_name 
    save_with_date = args.use_date 
    match_prob = args.match_prob 


# -

n_states = 2
n_actions = 2

all_population_size = 100 # number of random arms to generate
all_transitions = get_all_transitions(all_population_size)

all_transitions.shape

all_features = np.arange(all_population_size)

np.random.seed(seed)
random.seed(seed)
simulator = RMABSimulator(all_population_size, all_features, all_transitions,
            n_arms, episode_len, n_epochs, n_episodes, budget, number_states=n_states, reward_style='match',match_probability=match_prob)

import logging
logging.disable(logging.CRITICAL)

lamb_list = [0,1,2,4,6,8,12,16,24,32,48,64] 
lamb_list = [i/n_arms for i in lamb_list]

# ## Heterogenous Match Probability

np.random.seed(seed)
match_probabilities = [random.random() for i in range(all_population_size)]
simulator.match_probability = 0.5
simulator.match_probability_list = match_probabilities

if is_jupyter:
    lamb = 1
    np.random.seed(seed)
    random.seed(seed)
    greedy_reward = greedy_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    greedy_active_rate = simulator.total_active/(greedy_reward.size*n_arms)
    print(np.mean(greedy_reward) + lamb*n_arms*greedy_active_rate)

if is_jupyter:
    lamb = 1
    np.random.seed(seed)
    random.seed(seed)
    greedy_iterative_reward = greedy_iterative_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    greedy_iterative_active_rate = simulator.total_active/(greedy_iterative_reward.size*n_arms)
    print(np.mean(greedy_iterative_reward) + lamb*n_arms*greedy_active_rate)

if is_jupyter:
    lamb = 1
    np.random.seed(seed)
    random.seed(seed)
    greedy_iterative_q_reward = greedy_iterative_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb,use_Q=True)
    greedy_iterative_q_active_rate = simulator.total_active/(greedy_iterative_q_reward.size*n_arms)
    print(np.mean(greedy_iterative_q_reward) + lamb*n_arms*greedy_iterative_q_active_rate)

if is_jupyter:
    lamb = 1
    np.random.seed(seed)
    random.seed(seed)
    greedy_iterative_shapley_reward = greedy_iterative_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb,use_shapley=True)
    greedy_iterative_shapley_active_rate = simulator.total_active/(greedy_iterative_shapley_reward.size*n_arms)
    print(np.mean(greedy_iterative_shapley_reward) + lamb*n_arms*greedy_iterative_shapley_active_rate)

if is_jupyter:
    lamb = 1
    np.random.seed(seed)
    random.seed(seed)
    greedy_iterative_shapley_q_reward = greedy_iterative_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb,use_Q=True,use_shapley=True)
    greedy_iterative_shapley_q_active_rate = simulator.total_active/(greedy_iterative_shapley_q_reward.size*n_arms)
    print(np.mean(greedy_iterative_shapley_q_reward) + lamb*n_arms*greedy_iterative_shapley_q_active_rate)

if is_jupyter:
    np.random.seed(seed)
    random.seed(seed)
    approximate_combined_reward = optimal_whittle(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    approximate_combined_active_rate = simulator.total_active/(approximate_combined_reward.size*n_arms)
    print(np.mean(approximate_combined_reward) + lamb*n_arms*approximate_combined_active_rate)

if is_jupyter:
    lamb = 1
    np.random.seed(seed)
    random.seed(seed)
    mcts_reward = mcts_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    mcts_active_rate = simulator.total_active/(mcts_reward.size*n_arms)
    print(np.mean(mcts_reward) + lamb*n_arms*mcts_active_rate)

if is_jupyter:
    lamb = 1
    np.random.seed(seed)
    random.seed(seed)
    mcts_q_reward = mcts_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb,use_Q=True)
    mcts_q_active_rate = simulator.total_active/(mcts_q_reward.size*n_arms)
    print(np.mean(mcts_q_reward) + lamb*n_arms*mcts_q_active_rate)

if is_jupyter:
    np.random.seed(seed)
    random.seed(seed)
    optimal_reward = optimal_q_iteration(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    optimal_active_rate = simulator.total_active/(optimal_reward.size*n_arms)
    print(np.mean(optimal_reward) + lamb*n_arms*optimal_active_rate)

# +
greedy_reward_list = []
greedy_active_rate_list = []

for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    greedy_reward = greedy_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    greedy_active_rate = simulator.total_active/(greedy_reward.size*n_arms)
    greedy_reward_list.append(np.mean(greedy_reward))
    greedy_active_rate_list.append(greedy_active_rate)

# +
greedy_iterative_reward_list = []
greedy_iterative_active_rate_list = []

for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    greedy_iterative_reward = greedy_iterative_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    greedy_iterative_active_rate = simulator.total_active/(greedy_iterative_reward.size*n_arms)
    greedy_iterative_reward_list.append(np.mean(greedy_iterative_reward))
    greedy_iterative_active_rate_list.append(greedy_iterative_active_rate)

# -

greedy_iterative_q_reward_list = []
greedy_iterative_q_active_rate_list = []
for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    greedy_iterative_q_reward = greedy_iterative_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb,use_Q=True)
    greedy_iterative_q_active_rate = simulator.total_active/(greedy_iterative_q_reward.size*n_arms)
    greedy_iterative_q_reward_list.append(np.mean(greedy_iterative_q_reward))
    greedy_iterative_q_active_rate_list.append(greedy_iterative_q_active_rate)

# +
greedy_iterative_shapley_reward_list = []
greedy_iterative_shapley_active_rate_list = []

for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    greedy_iterative_shapley_reward = greedy_iterative_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb,use_shapley=True)
    greedy_iterative_shapley_active_rate = simulator.total_active/(greedy_iterative_shapley_reward.size*n_arms)
    greedy_iterative_shapley_reward_list.append(np.mean(greedy_iterative_shapley_reward))
    greedy_iterative_shapley_active_rate_list.append(greedy_iterative_shapley_active_rate)

# +
greedy_iterative_shapley_q_reward_list = []
greedy_iterative_shapley_q_active_rate_list = []

for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    greedy_iterative_shapley_q_reward = greedy_iterative_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb,use_Q=True,use_shapley=True)
    greedy_iterative_shapley_q_active_rate = simulator.total_active/(greedy_iterative_shapley_q_reward.size*n_arms)
    greedy_iterative_shapley_q_reward_list.append(np.mean(greedy_iterative_shapley_q_reward))
    greedy_iterative_shapley_q_active_rate_list.append(greedy_iterative_shapley_q_active_rate)

# +
approximate_combined_reward_list = []
approximate_combined_active_rate_list = []

for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    approximate_combined_reward = optimal_whittle(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    approximate_combined_active_rate = simulator.total_active/(approximate_combined_reward.size*n_arms)
    approximate_combined_reward_list.append(np.mean(approximate_combined_reward))
    approximate_combined_active_rate_list.append(approximate_combined_active_rate)

# +
mcts_reward_list = []
mcts_active_rate_list = []

for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    mcts_reward = mcts_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    mcts_active_rate = simulator.total_active/(mcts_reward.size*n_arms)
    mcts_reward_list.append(np.mean(mcts_reward))
    mcts_active_rate_list.append(mcts_active_rate)

# +
mcts_q_reward_list = []
mcts_q_active_rate_list = []

for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    mcts_q_reward = mcts_policy(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb,use_Q=True)
    mcts_q_active_rate = simulator.total_active/(mcts_q_reward.size*n_arms)
    mcts_q_reward_list.append(np.mean(mcts_q_reward))
    mcts_q_active_rate_list.append(mcts_q_active_rate)
# -

# ## Write Results

data = {
    'whittle_match': approximate_combined_reward_list, 
    'whittle_active': approximate_combined_active_rate_list,
    'greedy_match': greedy_reward_list, 
    'greedy_active': greedy_active_rate_list,
    'iterative_match': greedy_iterative_reward_list,
    'iterative_active': greedy_iterative_active_rate_list, 
    'iterative_q_match': greedy_iterative_q_reward_list, 
    'iterative_q_active': greedy_iterative_q_active_rate_list, 
    'iterative_shapley_match': greedy_iterative_shapley_reward_list, 
    'iterative_shapley_active': greedy_iterative_shapley_active_rate_list, 
    'iterative_q_shapley_match': greedy_iterative_shapley_q_reward_list,
    'iterative_q_shapley_active': greedy_iterative_shapley_q_active_rate_list,
    'mcts_match': mcts_reward_list,
    'mcts_active': mcts_active_rate_list, 
    'mcts_q_match': mcts_q_reward_list, 
    'mcts_q_active': mcts_q_active_rate_list,
    'parameters': 
        {'seed'      : seed,
        'n_arms'    : n_arms,
        'budget'    : budget,
        'discount'  : discount, 
        'alpha'     : alpha, 
        'n_episodes': n_episodes, 
        'episode_len': episode_len, 
        'n_epochs'  : n_epochs, 
        'match_prob': match_prob, 
        'lambda_list': lamb_list,} 
}

if n_arms <= 6:
    np.random.seed(seed)
    random.seed(seed)
    _ = optimal_q_iteration(simulator, n_episodes, n_epochs, discount,reward_function='activity')
    optimal_active_rate = simulator.total_active/(_.size*n_arms)

    np.random.seed(seed)
    random.seed(seed)
    optimal_match_reward = optimal_q_iteration(simulator, n_episodes, n_epochs, discount)

    joint_match = []
    joint_active = []

    for lamb in lamb_list:
        np.random.seed(seed)
        random.seed(seed)
        joint_combined_reward = optimal_q_iteration(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
        joint_combined_active_rate = simulator.total_active/(joint_combined_reward.size*n_arms)

        joint_match.append(np.mean(joint_combined_reward))
        joint_active.append(joint_combined_active_rate)
    
    data['joint_match'] = joint_match 
    data['joint_active'] = joint_active 
    data['optimal_match'] = np.mean(optimal_match_reward)
    data['optimal_active'] = optimal_active_rate

save_path = get_save_path('combined',save_name,seed,use_date=save_with_date)

delete_duplicate_results('combined',save_name,data)

json.dump(data,open('../results/'+save_path,'w'))


