# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: food
#     language: python
#     name: python3
# ---

# # All Policies

# Analyze the performance of our Whittle and Adaptive Policies

# %load_ext autoreload
# %autoreload 2

import numpy as np
import random 
import matplotlib.pyplot as plt
import json 
import argparse 
import sys
import secrets
from itertools import combinations

from rmab.simulator import run_multi_seed, get_contextual_probabilities
from rmab.whittle_policies import *
from rmab.baseline_policies import *
from rmab.mcts_policies import *
from rmab.utils import get_save_path, delete_duplicate_results, restrict_resources

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 43
    n_arms      = 10
    volunteers_per_arm = 1
    budget      = 5
    discount    = 0.9999
    alpha       = 3 
    n_episodes  = 5
    episode_len = 50
    n_epochs    = 1
    save_with_date = False 
    lamb = 0
    prob_distro = 'uniform'
    reward_type = "probability"
    reward_parameters = {'universe_size': 20, 'arm_set_low': 0, 'arm_set_high': 1, 'recovery_rate': 0}
    out_folder = ''
    time_limit = 100
    run_rate_limits = False 
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_arms',         '-N', help='num beneficiaries (arms)', type=int, default=2)
    parser.add_argument('--volunteers_per_arm',         '-V', help='volunteers per arm', type=int, default=5)
    parser.add_argument('--episode_len',    '-H', help='episode length', type=int, default=50)
    parser.add_argument('--n_episodes',     '-T', help='num episodes', type=int, default=105)
    parser.add_argument('--budget',         '-B', help='budget', type=int, default=3)
    parser.add_argument('--n_epochs',       '-E', help='number of epochs (num_repeats)', type=int, default=1)
    parser.add_argument('--discount',       '-d', help='discount factor', type=float, default=0.9)
    parser.add_argument('--alpha',          '-a', help='alpha: for conf radius', type=float, default=3)
    parser.add_argument('--lamb',          '-l', help='lambda for matching-engagement tradeoff', type=float, default=0.5)
    parser.add_argument('--universe_size', help='For set cover, total num unvierse elems', type=int, default=10)
    parser.add_argument('--recovery_rate', help='How fast volunteers recover', type=float, default=0.1)
    parser.add_argument('--arm_set_low', help='Least size of arm set, for set cover', type=float, default=3)
    parser.add_argument('--arm_set_high', help='Largest size of arm set, for set cover', type=float, default=6)
    parser.add_argument('--reward_type',          '-r', help='Which type of custom reward', type=str, default='set_cover')
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=42)
    parser.add_argument('--prob_distro',           '-p', help='which prob distro [uniform,uniform_small,uniform_large,normal]', type=str, default='uniform')
    parser.add_argument('--out_folder', help='Which folder to write results to', type=str, default='iterative')
    parser.add_argument('--time_limit', help='Online time limit for computation', type=float, default=100)
    parser.add_argument('--use_date', action='store_true')
    parser.add_argument('--run_rate_limits', action='store_true')

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
    lamb = args.lamb
    save_with_date = args.use_date
    prob_distro = args.prob_distro
    out_folder = args.out_folder
    reward_type = args.reward_type
    run_rate_limits = args.run_rate_limits
    recovery_rate = args.recovery_rate 
    reward_parameters = {'universe_size': args.universe_size,
                        'arm_set_low': args.arm_set_low, 
                        'arm_set_high': args.arm_set_high, 
                        'recovery_rate': args.recovery_rate}
    time_limit = args.time_limit 

save_name = secrets.token_hex(4)  
# -

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
        'prob_distro': prob_distro, 
        'reward_type': reward_type, 
        'universe_size': reward_parameters['universe_size'],
        'arm_set_low': reward_parameters['arm_set_low'], 
        'arm_set_high': reward_parameters['arm_set_high'],
        'time_limit': time_limit, 
        'recovery_rate': reward_parameters['recovery_rate']
        } 

results['parameters']

# ## Index Policies

seed_list = [seed]

# +
policy = greedy_policy
name = "greedy"
print(name)

rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']
results['ratio'] = simulator.ratio 
print(np.mean(rewards['reward']))

# +
policy = random_policy
name = "random"
print(name)

rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']
print(np.mean(rewards['reward']))
# -

if n_arms * volunteers_per_arm <= 4 and 'multi_state' not in prob_distro and 'two_timescale' not in prob_distro:
    policy = q_iteration_policy
    per_epoch_function = q_iteration_custom_epoch()
    name = "optimal"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function=per_epoch_function,test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']
    print(np.mean(rewards['reward']))

# +
policy = whittle_activity_policy
name = "whittle_activity"
print(name)

rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']
print(np.mean(rewards['reward']))

# +
policy = whittle_policy
name = "linear_whittle"
print(name)

rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']
print(np.mean(rewards['reward']))
# -

if 'context' in reward_type and n_arms * volunteers_per_arm <= 250:
    policy = fast_contextual_whittle_policy
    name = "fast_contextual_whittle"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))

if n_arms * volunteers_per_arm <= 250 and 'context' in reward_type  and 'two_timescale' not in prob_distro:
    policy = contextual_whittle_policy
    name = "contextual_whittle"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))

if n_arms * volunteers_per_arm <= 1000:
    policy = shapley_whittle_custom_policy 
    name = "shapley_whittle_custom"
    print(name)

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50),shapley_iterations=1000)
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']
    print(np.mean(rewards['reward']))

if n_arms * volunteers_per_arm <= 250 and 'context' in reward_type and episode_len<=10000:
    policy = fast_contextual_shapley_policy
    name = "fast_contextual_shapley"
    print(name)

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))

if n_arms * volunteers_per_arm <= 250 and 'context' in reward_type and 'two_timescale' not in prob_distro:
    policy = contextual_shapley_policy
    name = "contextual_shapley"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))

if n_arms * volunteers_per_arm <= 250 and episode_len<=10000:
    policy = whittle_iterative_policy
    name = "iterative_whittle"
    print(name)

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']

    print(np.mean(rewards['reward']))

if n_arms * volunteers_per_arm <= 250 and 'context' in reward_type and episode_len <= 10000:
    policy = non_contextual_whittle_iterative_policy
    name = "non_contextual_iterative_whittle"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']

    print(np.mean(rewards['reward']))


if n_arms * volunteers_per_arm <= 25 and reward_type != 'probability_context' and episode_len<=10000:
    policy = shapley_whittle_iterative_policy
    name = "shapley_iterative_whittle"
    print(name)

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50),shapley_iterations=1000)
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']
    print(np.mean(rewards['reward']))

if n_arms * volunteers_per_arm <= 250 and 'context' in reward_type and episode_len <= 10000:
    policy = non_contextual_shapley_whittle_iterative_policy
    name = "non_contextual_shapley_iterative_whittle"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']

    print(np.mean(rewards['reward']))


if n_arms * volunteers_per_arm <= 25 and 'two_timescale' not in prob_distro and 'multi_state' not in prob_distro:
    policy = mcts_linear_policy
    name = "mcts_linear"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50),mcts_test_iterations=400)
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']
    print(np.mean(rewards['reward']))


if n_arms * volunteers_per_arm <= 250 and 'two_timescale' not in prob_distro and 'context' in reward_type:
    policy = non_contextual_mcts_linear_policy
    name = "non_contextual_mcts_linear"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']

    print(np.mean(rewards['reward']))


if n_arms * volunteers_per_arm <= 25 and 'two_timescale' not in prob_distro and 'multi_state' not in prob_distro:
    policy = mcts_shapley_policy
    name = "mcts_shapley"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50),mcts_test_iterations=400)
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']

    print(np.mean(rewards['reward']))


if n_arms * volunteers_per_arm <= 250 and 'two_timescale' not in prob_distro and 'context' in reward_type:
    policy = non_contextual_mcts_shapley_policy
    name = "non_contextual_mcts_shapley"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    results['{}_burned_out_rate'.format(name)] =  rewards['burned_out_rate']

    print(np.mean(rewards['reward']))


if n_arms * volunteers_per_arm <= 25 and run_rate_limits:
    policy = mcts_shapley_policy
    name = "mcts_shapley_40"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50),mcts_test_iterations=40)
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))


if n_arms * volunteers_per_arm <= 25 and run_rate_limits:
    policy = mcts_shapley_policy
    name = "mcts_shapley_4"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50),mcts_test_iterations=4)
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))


if n_arms * volunteers_per_arm <= 50 and run_rate_limits:
    policy = shapley_whittle_iterative_policy
    name = "shapley_iterative_whittle_100"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50),shapley_iterations=100)
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))

if n_arms * volunteers_per_arm <= 50 and run_rate_limits:
    policy = shapley_whittle_iterative_policy
    name = "shapley_iterative_whittle_10"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len,shapley_iterations=10)
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))

if n_arms * volunteers_per_arm <= 50 and run_rate_limits:
    policy = shapley_whittle_iterative_policy
    name = "shapley_iterative_whittle_1"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50),shapley_iterations=1)
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))

# ## Write Data

save_path = get_save_path(out_folder,save_name,seed,use_date=save_with_date)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))
