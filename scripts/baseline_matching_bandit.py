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

# %load_ext autoreload
# %autoreload 2

import numpy as np
import random 
import matplotlib.pyplot as plt
import json 
import argparse 
import sys
from itertools import combinations

from rmab.simulator import RMABSimulator
from rmab.baselines import optimal_policy, random_policy, optimal_match_slow_policy, myopic_match_n_step
from rmab.fr_dynamics import get_all_transitions
from rmab.utils import get_save_path, delete_duplicate_results

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 42
    n_arms      = 8
    budget      = 3
    discount    = 0.9
    alpha       = 3 
    n_episodes  = 30
    episode_len = 20
    n_epochs    = 10
    save_name = 'results'
    match_prob = 0.5
    save_with_date = True 
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
    parser.add_argument('--save_name',      '-n', help='save name', type=str, default='results')
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

simulator.reset_instance()
simulator.transitions.shape

np.random.seed(seed)
random.seed(seed)
random_rewards = random_policy(simulator, n_episodes, n_epochs)
random_active_rate = simulator.total_active/(random_rewards.size * n_arms)

np.random.seed(seed)
random.seed(seed)
optimal_reward = optimal_policy(simulator, n_episodes, n_epochs, discount)
optimal_active_rate = simulator.total_active/(optimal_reward.size*n_arms)

np.random.seed(seed)
random.seed(seed)
zero_step_rewards = myopic_match_n_step(simulator, n_episodes, n_epochs, discount,0)
zero_step_active_rate = simulator.total_active/(zero_step_rewards.size*n_arms)

np.random.seed(seed)
random.seed(seed)
one_step_rewards = myopic_match_n_step(simulator, n_episodes, n_epochs, discount,1)
one_step_active_rate = simulator.total_active/(one_step_rewards.size*n_arms)

np.random.seed(seed)
random.seed(seed)
infinite_step_rewards = myopic_match_n_step(simulator, n_episodes, n_epochs, discount,-1)
infinite_step_active_rate = simulator.total_active/(infinite_step_rewards.size*n_arms)

np.random.seed(seed)
random.seed(seed)
whittle_approximate_rewards = optimal_policy(simulator, n_episodes, n_epochs, discount,use_match_reward=True)
whittle_approximate_active_rate = simulator.total_active/(whittle_approximate_rewards.size*n_arms)

mean_rewards = {'random_rewards': np.mean(random_rewards), 
 'optimal_rewards': np.mean(optimal_reward), 
 'zero_step_rewards': np.mean(zero_step_rewards),
 'one_step_rewards': np.mean(one_step_rewards),
 'infinite_step_rewards': np.mean(infinite_step_rewards),
 'whittle_approximate_rewards': np.mean(whittle_approximate_rewards)}

active_rates = {'random_rewards': np.mean(random_active_rate), 
 'optimal_rewards': np.mean(optimal_active_rate), 
 'zero_step_rewards': np.mean(zero_step_active_rate),
 'one_step_rewards': np.mean(one_step_active_rate),
 'infinite_step_rewards': np.mean(infinite_step_active_rate),
 'whittle_approximate_rewards': np.mean(whittle_approximate_active_rate)}

std_rewards = {'random_rewards': np.std(random_rewards), 
 'optimal_rewards': np.std(optimal_reward), 
 'zero_step_rewards': np.std(zero_step_rewards),
 'one_step_rewards': np.std(one_step_rewards),
 'infinite_step_rewards': np.std(infinite_step_rewards),
 'whittle_approximate_rewards': np.std(whittle_approximate_rewards), 
 }

print("N arms {}".format(n_arms))

if n_arms <= 6:
    np.random.seed(seed)
    random.seed(seed)
    optimal_match_rewards = optimal_match_slow_policy(simulator, n_episodes, n_epochs, discount)
    optimal_match_active_rate = simulator.total_active/(optimal_match_rewards.size*n_arms)

    mean_rewards['optimal_match_rewards'] = np.mean(optimal_match_rewards)
    active_rates['optimal_match_rewards'] = np.mean(optimal_match_active_rate)
    std_rewards['optimal_match_rewards'] = np.std(optimal_match_rewards)

mean_rewards

data = {
    'mean_reward': mean_rewards, 
    'std_reward': std_rewards,
    'active_rate': active_rates, 
    'parameters': 
        {'seed'      : seed,
        'n_arms'    : n_arms,
        'budget'    : budget,
        'discount'  : discount, 
        'alpha'     : alpha, 
        'n_episodes': n_episodes, 
        'episode_len': episode_len, 
        'n_epochs'  : n_epochs, 
        'match_prob': match_prob} 
}

save_path = get_save_path('matching_baseline',save_name,seed,use_date=save_with_date)

delete_duplicate_results('matching_baseline',save_name,data)

json.dump(data,open('../results/'+save_path,'w'))
