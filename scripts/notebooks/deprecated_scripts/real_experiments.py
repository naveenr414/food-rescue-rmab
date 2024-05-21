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

# # Real Experiments

# Analyze the performance of various algorithms to solve the joint matching + activity task, when the number of volunteers is large and structured

# %load_ext autoreload
# %autoreload 2

import numpy as np
import random 
import matplotlib.pyplot as plt
import json 
import argparse 
import sys
import secrets

from rmab.simulator import RMABSimulator
from rmab.omniscient_policies import *
from rmab.fr_dynamics import get_all_transitions, get_match_probs, get_dict_match_probs
from rmab.utils import get_save_path, delete_duplicate_results

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 42
    n_arms      = 100
    volunteers_per_arm = 100
    budget      = 3
    discount    = 0.9
    alpha       = 3 
    n_episodes  = 30 
    episode_len = 20 
    n_epochs    = 1
    save_with_date = False 
    TIME_PER_RUN = 0.01 * 1000
    lamb = 1/(n_arms*volunteers_per_arm)
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=100)
    parser.add_argument('--volunteers_per_arm',         '-V', help='volunteers per arm', type=int, default=100)
    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=20)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=30)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=3)
    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=1)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.9)
    parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)
    parser.add_argument('--lamb',          '-l', help='lambda for matching-engagement tradeoff', type=float, default=1)
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=42)
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
    lamb = args.lamb /(volunteers_per_arm*n_arms)
    save_with_date = args.use_date
    TIME_PER_RUN = args.time_per_run

save_name = secrets.token_hex(4)  
# -

n_states = 2
n_actions = 2

all_population_size = 100 # number of random arms to generate
all_transitions = get_all_transitions(all_population_size)

random.seed(seed)
np.random.seed(seed)

# +
all_features = np.arange(all_population_size)

match_probabilities = get_match_probs([i//volunteers_per_arm+1 for i in range(all_population_size * volunteers_per_arm)])
# -

np.random.seed(seed)
random.seed(seed)
simulator = RMABSimulator(all_population_size, all_features, all_transitions,
            n_arms, volunteers_per_arm, episode_len, n_epochs, n_episodes, budget, discount,number_states=n_states, reward_style='match',match_probability_list=match_probabilities,TIME_PER_RUN=TIME_PER_RUN)

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

# ## Index Policies

# +
policy = greedy_policy
name = "greedy"
greedy_reward, greedy_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
time_greedy = simulator.time_taken
print(np.mean(greedy_reward) + lamb*n_arms*volunteers_per_arm*greedy_active_rate)

results['{}_match'.format(name)] = np.mean(greedy_reward) 
results['{}_active'.format(name)] = greedy_active_rate 
results['{}_time'.format(name)] = time_greedy 

# +
policy = random_policy
name = "random"
random_reward, random_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
time_random = simulator.time_taken
print(np.mean(random_reward) + random_active_rate*lamb*n_arms*volunteers_per_arm)

results['{}_match'.format(name)] = np.mean(random_reward)
results['{}_active'.format(name)] = random_active_rate 
results['{}_time'.format(name)] = time_random 

# +
policy = whittle_activity_policy
name = "whittle_engagement"
whittle_activity_reward, whittle_activity_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
time_whittle_activity = simulator.time_taken    
print(np.mean(whittle_activity_reward) + whittle_activity_active_rate*lamb*n_arms*volunteers_per_arm)

results['{}_match'.format(name)] = np.mean(whittle_activity_reward) 
results['{}_active'.format(name)] = whittle_activity_active_rate 
results['{}_time'.format(name)] = time_whittle_activity 

# +
policy = whittle_policy
name = "linear_whittle"
whittle_reward, whittle_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
time_whittle = simulator.time_taken    
print(np.mean(whittle_reward) + whittle_active_rate*lamb*n_arms*volunteers_per_arm)

results['{}_match'.format(name)] = np.mean(whittle_reward) 
results['{}_active'.format(name)] = whittle_active_rate 
results['{}_time'.format(name)] = time_whittle 

# +
policy = shapley_whittle_policy 
name = "shapley_whittle"
whittle_shapley_reward, whittle_shapley_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
time_whittle_shapley = simulator.time_taken
print(np.mean(whittle_shapley_reward) + whittle_shapley_active_rate*lamb*n_arms*volunteers_per_arm)

results['{}_match'.format(name)] = np.mean(whittle_shapley_reward) 
results['{}_active'.format(name)] = whittle_shapley_active_rate 
results['{}_time'.format(name)] = time_whittle_shapley 

# +
policy = whittle_greedy_policy
name = "whittle_greedy"
whittle_greedy_reward, whittle_greedy_active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb)
time_whitte_greedy = simulator.time_taken
print(np.mean(whittle_greedy_reward) + whittle_greedy_active_rate*lamb*n_arms*volunteers_per_arm)

results['{}_match'.format(name)] = np.mean(whittle_greedy_reward)
results['{}_active'.format(name)] = whittle_greedy_active_rate 
results['{}_time'.format(name)] = time_whitte_greedy 
# -

# ## Write Data

save_path = get_save_path('real_data',save_name,seed,use_date=save_with_date)

delete_duplicate_results('real_data',"",results)

json.dump(results,open('../results/'+save_path,'w'))


