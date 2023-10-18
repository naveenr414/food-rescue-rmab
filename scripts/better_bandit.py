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
from scipy.stats import norm

# +
from rmab.simulator import RMABSimulator, random_valid_transition, random_valid_transition_round_down, synthetic_transition_small_window
from rmab.uc_whittle import UCWhittle, UCWhittleFixed, QP_step,UCB_step
from rmab.utils import Memoizer, get_ucb_conf

from rmab.ucw_value import UCWhittle_value, UCWhittle_value_fixed
from rmab.baselines import optimal_policy, random_policy, WIQL
from rmab.fr_dynamics import get_all_transitions
from rmab.utils import get_save_path, delete_duplicate_results

# -

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
    save_name = 'normal_confidence'
    save_with_date = False 
    dataset = 'fr'
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
    parser.add_argument('--dataset',      '-ds', help='which dataset', type=str, default='fr')
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
    dataset = args.dataset 


# -

n_states = 2
n_actions = 2

# +
all_population_size = 100 # number of random arms to generate

if dataset == 'fr':
    all_transitions = get_all_transitions(all_population_size)
elif dataset == 'synthetic':
    all_transitions = random_valid_transition(all_population_size, n_states, n_actions)
else:
    raise Exception("Dataset {} is not found".format(dataset))
# -

all_features = np.arange(all_population_size)

np.random.seed(seed)
random.seed(seed)
simulator = RMABSimulator(all_population_size, all_features, all_transitions,
            n_arms, episode_len, n_epochs, n_episodes, budget, number_states=n_states)

np.random.seed(seed)
random.seed(seed)
rewards_without_norm = UCWhittleFixed(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB',norm_confidence=False)

np.random.seed(seed)
random.seed(seed)
rewards_with_norm = UCWhittleFixed(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB',norm_confidence=True)

data = {
    'mean_reward_baseline': np.mean(rewards_without_norm), 
    'mean_reward_norm': np.mean(rewards_with_norm), 
    'parameters': 
        {'seed'      : seed,
        'n_arms'    : n_arms,
        'budget'    : budget,
        'discount'  : discount, 
        'alpha'     : alpha, 
        'n_episodes': n_episodes, 
        'episode_len': episode_len, 
        'n_epochs'  : n_epochs, 
        'dataset': dataset} 
}

save_path = get_save_path('better_bandit',save_name,seed,use_date=save_with_date)

delete_duplicate_results('better_bandit',save_name,data)

json.dump(data,open('../results/'+save_path,'w'))
