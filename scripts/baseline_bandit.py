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

from rmab.simulator import RMABSimulator, random_valid_transition, random_valid_transition_round_down, synthetic_transition_small_window
from rmab.uc_whittle import UCWhittle
from rmab.ucw_value import UCWhittle_value
from rmab.baselines import optimal_policy, random_policy, WIQL
from rmab.fr_dynamics import get_all_transitions
from rmab.utils import get_save_path, delete_duplicate_results


is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 43
    n_arms      = 8
    budget      = 3
    discount    = 0.9
    alpha       = 3 
    n_episodes  = 30
    episode_len = 20
    n_epochs    = 10
    save_name = 'hyperparameter'
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


# -

n_states = 2
n_actions = 2

all_population_size = 100 # number of random arms to generate
all_transitions = get_all_transitions(all_population_size)

all_transitions.shape

all_features = np.arange(all_population_size)

simulator = RMABSimulator(all_population_size, all_features, all_transitions,
            n_arms, episode_len, n_epochs, n_episodes, budget, number_states=n_states)

np.random.seed(seed)
random.seed(seed)
random_rewards = random_policy(simulator, n_episodes, n_epochs)

np.random.seed(seed)
random.seed(seed)
optimal_reward = optimal_policy(simulator, n_episodes, n_epochs, discount)

np.random.seed(seed)
random.seed(seed)
wiql_rewards = WIQL(simulator, n_episodes, n_epochs)

np.random.seed(seed)
random.seed(seed)
ucw_extreme_rewards = UCWhittle(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='extreme')

np.random.seed(seed)
random.seed(seed)
ucw_ucb_rewards = UCWhittle(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB')

mean_rewards = {'random_rewards': np.mean(random_rewards), 
 'optimal_rewards': np.mean(optimal_reward), 
 'wiql_rewards': np.mean(wiql_rewards), 
 'extreme_rewards': np.mean(ucw_extreme_rewards), 
 'ucb_rewards': np.mean(ucw_ucb_rewards)}

std_rewards = {'random_rewards': np.std(random_rewards), 
 'optimal_rewards': np.std(optimal_reward), 
 'wiql_rewards': np.std(wiql_rewards), 
 'extreme_rewards': np.std(ucw_extreme_rewards), 
 'ucb_rewards': np.std(ucw_ucb_rewards)}

random_match = 1-np.sum(random_rewards == 0)/random_rewards.size
optimal_match = 1-np.sum(optimal_reward == 0)/optimal_reward.size 
wiql_match = 1-np.sum(wiql_rewards == 0)/wiql_rewards.size 
ucw_extreme_match = 1-np.sum(ucw_extreme_rewards == 0)/ucw_extreme_rewards.size 
ucw_ucb_match = 1-np.sum(ucw_ucb_rewards == 0)/ucw_ucb_rewards.size

match_rates = {
    'random_match': random_match, 
    'optimal_match': optimal_match, 
    'wiql_match': wiql_match, 
    'extreme_match': ucw_extreme_match, 
    'ucb_match': ucw_ucb_match, 
}
match_rates

data = {
    'mean_reward': mean_rewards, 
    'std_reward': std_rewards,
    'match_rate': match_rates, 
    'parameters': 
        {'seed'      : seed,
        'n_arms'    : n_arms,
        'budget'    : budget,
        'discount'  : discount, 
        'alpha'     : alpha, 
        'n_episodes': n_episodes, 
        'episode_len': episode_len, 
        'n_epochs'  : n_epochs} 
}

save_path = get_save_path('baseline',save_name,seed,use_date=save_with_date)

delete_duplicate_results('baseline',save_name,data)

json.dump(data,open('../results/'+save_path,'w'))
