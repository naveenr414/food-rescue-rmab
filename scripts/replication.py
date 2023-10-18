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
from rmab.uc_whittle import UCWhittle, UCWhittleFixed 
from rmab.ucw_value import UCWhittle_value
from rmab.baselines import optimal_policy, random_policy, WIQL
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
all_transitions = random_valid_transition(all_population_size, n_states, n_actions)

all_transitions.shape

all_features = np.arange(all_population_size)

np.random.seed(seed)
random.seed(seed)
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
ucw_value_rewards = UCWhittle_value(simulator, n_episodes, n_epochs, discount, alpha=alpha)

np.random.seed(seed)
random.seed(seed)
ucw_ucb_rewards = UCWhittle(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB')

np.random.seed(seed)
random.seed(seed)
ucw_qp_rewards = UCWhittle(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='QP')

np.random.seed(seed)
random.seed(seed)
ucw_ucb_rewards_fixed = UCWhittleFixed(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB')

x_vals = np.arange(n_episodes * episode_len + 1)


def get_cum_sum(reward):
        cum_sum = reward.cumsum(axis=1).mean(axis=0)
        cum_sum = cum_sum / (x_vals + 1)
        return smooth(cum_sum)
def smooth(rewards, weight=0.7):
    """ smoothed exponential moving average """
    prev = rewards[0]
    smoothed = np.zeros(len(rewards))
    for i, val in enumerate(rewards):
        smoothed_val = prev * weight + (1 - weight) * val
        smoothed[i] = smoothed_val
        prev = smoothed_val

    return smoothed


use_algos = ['optimal', 'ucw_ucb', 'ucw_qp', 'ucw_extreme', 'wiql', 'random', 'fixed']
rewards = {
    'optimal': optimal_reward,
    'ucw_qp': ucw_qp_rewards, 
    'ucw_extreme': ucw_extreme_rewards, 
    'wiql': wiql_rewards, 
    'random': random_rewards, 
    'fixed': ucw_ucb_rewards_fixed,  
    'ucw_ucb': ucw_ucb_rewards, 
}
colors = {'optimal': 'purple', 'ucw_value': 'b', 'ucw_qp': 'c', 'ucw_qp_min': 'goldenrod', 'ucw_ucb': 'darkorange',
                'ucw_extreme': 'r', 'wiql': 'limegreen', 'random': 'brown', 'fixed': 'goldenrod'}

# +
cum_sum_by_algo = {}
for algo in use_algos:
    cum_sum_by_algo[algo] = get_cum_sum(rewards[algo])

mean_by_algo = {}
for algo in use_algos: 
    mean_by_algo[algo] = rewards[algo].mean(axis=0)
# -

for algo in use_algos:
    plt.plot(x_vals, get_cum_sum(rewards[algo]), c=colors[algo], label=algo)
plt.legend()

for algo in use_algos:
    plt.plot(x_vals, smooth(rewards[algo].mean(axis=0)), c=colors[algo], label=algo)
plt.legend()

for i in cum_sum_by_algo:
    cum_sum_by_algo[i] = cum_sum_by_algo[i].tolist() 
    mean_by_algo[i] = mean_by_algo[i].tolist() 

data = {
    'cum_sum': cum_sum_by_algo,
    'mean_rewards': mean_by_algo,  
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

save_path = get_save_path('replication',save_name,seed,use_date=False)

delete_duplicate_results('replication',save_name,data)

json.dump(data,open('../results/'+save_path,'w'))


