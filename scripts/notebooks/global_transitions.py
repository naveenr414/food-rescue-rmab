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

# # Global Transitions

# Analyze the performance of our Linear Programming and Individual Global Transition Policies

# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append('/usr0/home/naveenr/projects/food_rescue_preferences')

import numpy as np
import random 
import matplotlib.pyplot as plt
import json 
import argparse 
import sys
from copy import deepcopy
import secrets
from rmab.simulator import run_multi_seed
from rmab.whittle_policies import q_iteration_policy, whittle_activity_policy, shapley_whittle_custom_policy
from rmab.baseline_policies import random_policy
from rmab.utils import restrict_resources
from rmab.global_policies import *
from rmab.utils import get_save_path, delete_duplicate_results, restrict_resources


is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 43
    n_arms      = 20
    volunteers_per_arm = 1
    budget      = 5
    discount    = 0.9
    alpha       = 3 
    n_episodes  = 5
    episode_len = 50 
    n_epochs    = 1
    save_with_date = False 
    lamb = 0
    prob_distro = 'global_transition_extreme'
    reward_type = "global_transition"
    reward_parameters = {'universe_size': 20, 'arm_set_low': 6, 'arm_set_high': 8}
    out_folder = 'unknown_transitions/baseline_policies'
    time_limit = 100
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
    parser.add_argument('--arm_set_low', help='Least size of arm set, for set cover', type=float, default=3)
    parser.add_argument('--arm_set_high', help='Largest size of arm set, for set cover', type=float, default=6)
    parser.add_argument('--reward_type',          '-r', help='Which type of custom reward', type=str, default='set_cover')
    parser.add_argument('--seed',           '-s', help='random seed', type=int, default=42)
    parser.add_argument('--prob_distro',           '-p', help='which prob distro [uniform,uniform_small,uniform_large,normal]', type=str, default='uniform')
    parser.add_argument('--out_folder', help='Which folder to write results to', type=str, default='iterative')
    parser.add_argument('--time_limit', help='Online time limit for computation', type=float, default=100)
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
    lamb = args.lamb
    save_with_date = args.use_date
    prob_distro = args.prob_distro
    out_folder = args.out_folder
    reward_type = args.reward_type
    reward_parameters = {'universe_size': args.universe_size,
                        'arm_set_low': args.arm_set_low, 
                        'arm_set_high': args.arm_set_high}
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
        } 

# ## Global Transition Policies

seed_list = [seed]
restrict_resources()

# +
policy = random_policy
name = "random"

rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = optimistic_policy
name = "individual_optimistic"

rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = average_policy
name = "individual_average"

rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = pessimistic_policy
name = "individual_pessimistic"

rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = shapley_whittle_custom_policy
name = "shapley_whittle"

temp_parameters = deepcopy(results['parameters'])
temp_parameters['reward_type'] = 'probability'

rewards, memory, simulator = run_multi_seed(seed_list,policy,temp_parameters,test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = shapley_whittle_global_transition_custom_policy
name = "shapley_whittle_global_transition"

temp_parameters = deepcopy(results['parameters'])
temp_parameters['reward_type'] = 'probability'

rewards, memory, simulator = run_multi_seed(seed_list,policy,temp_parameters,test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = whittle_activity_policy
name = "whittle_activity"

rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = global_transition_linear_policy
name = "linear_program"

rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))
# -

if n_arms * volunteers_per_arm <= 4:
    policy = q_iteration_policy
    per_epoch_function = q_iteration_custom_epoch_global_transition()
    name = "optimal"

    rewards, memory, simulator = run_multi_seed(seed_list,policy,results['parameters'],per_epoch_function=per_epoch_function,test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))

# ## Write Data

save_path = get_save_path(out_folder,save_name,seed,use_date=save_with_date)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))
