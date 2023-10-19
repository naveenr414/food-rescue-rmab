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
from rmab.uc_whittle import UCWhittle, UCWhittleFixed, UCWhittleMatch, NormPlusMatch
from rmab.ucw_value import UCWhittle_value
from rmab.baselines import optimal_policy, random_policy, WIQL, optimal_match_policy
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
    save_name = 'results_all_agents'
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

np.random.seed(seed)
random.seed(seed)
random_rewards = random_policy(simulator, n_episodes, n_epochs)
random_active_rate = simulator.total_active/(random_rewards.size * n_arms)

np.random.seed(seed)
random.seed(seed)
optimal_match_reward = optimal_match_policy(simulator, n_episodes, n_epochs, discount)
optimal_match_active_rate = simulator.total_active/(optimal_match_reward.size*n_arms)

np.random.seed(seed)
random.seed(seed)
optimal_reward = optimal_policy(simulator, n_episodes, n_epochs, discount)
optimal_active_rate = simulator.total_active/(optimal_reward.size*n_arms)

np.random.seed(seed)
random.seed(seed)
wiql_rewards = WIQL(simulator, n_episodes, n_epochs)
wiql_active_rate = simulator.total_active/(wiql_rewards.size*n_arms)

np.random.seed(seed)
random.seed(seed)
ucw_extreme_rewards = UCWhittle(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='extreme')
ucw_extreme_active_rate = simulator.total_active/(ucw_extreme_rewards.size*n_arms)

np.random.seed(seed)
random.seed(seed)
ucw_ucb_rewards = UCWhittle(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB')
ucw_ucb_active_rate = simulator.total_active/(ucw_ucb_rewards.size*n_arms)

np.random.seed(seed)
random.seed(seed)
rewards_without_norm = UCWhittleFixed(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB',norm_confidence=False)
ucw_without_norm_active_rate = simulator.total_active/(rewards_without_norm.size*n_arms)

np.random.seed(seed)
random.seed(seed)
rewards_with_norm = UCWhittleFixed(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB',norm_confidence=True)
ucw_with_norm_active_rate = simulator.total_active/(rewards_with_norm.size*n_arms)

np.random.seed(seed)
random.seed(seed)
rewards_match_heuristic = UCWhittleMatch(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB')
match_heuristic_active_rate = simulator.total_active/(rewards_match_heuristic.size*n_arms)

np.random.seed(seed)
random.seed(seed)
rewards_combined = NormPlusMatch(simulator, n_episodes, n_epochs, discount, alpha=alpha, method='UCB',lamb=0.25)
combined_active_rate = simulator.total_active/(rewards_combined.size*n_arms)

mean_rewards = {'random_rewards': np.mean(random_rewards), 
 'optimal_rewards': np.mean(optimal_reward), 
 'optimal_match_rewards': np.mean(optimal_match_reward),
 'wiql_rewards': np.mean(wiql_rewards), 
 'extreme_rewards': np.mean(ucw_extreme_rewards), 
 'ucb_rewards': np.mean(ucw_ucb_rewards), 
 'fixed_rewards': np.mean(rewards_without_norm), 
 'norm_rewards': np.mean(rewards_with_norm), 
 'predicted_optimal_match_rewards': np.mean(rewards_match_heuristic), 
 'combined_rewards': np.mean(rewards_combined)}

active_rates = {'random_rewards': random_active_rate, 
 'optimal_rewards': optimal_active_rate, 
 'optimal_match_rewards': optimal_match_active_rate,
 'wiql_rewards': wiql_active_rate, 
 'extreme_rewards': ucw_extreme_active_rate, 
 'ucb_rewards': ucw_ucb_active_rate, 
 'fixed_rewards': ucw_without_norm_active_rate, 
 'norm_rewards': ucw_with_norm_active_rate, 
 'predicted_optimal_match_rewards': match_heuristic_active_rate, 
 'combined_rewards': combined_active_rate}

mean_rewards 

std_rewards = {'random_rewards': np.std(random_rewards), 
 'optimal_rewards': np.std(optimal_reward), 
 'optimal_match_reward': np.mean(optimal_match_reward), 
 'wiql_rewards': np.std(wiql_rewards), 
 'extreme_rewards': np.std(ucw_extreme_rewards), 
 'ucb_rewards': np.std(ucw_ucb_rewards), 
 'fixed_rewards': np.std(rewards_without_norm), 
 'norm_rewards': np.std(rewards_with_norm), 
 'predicted_optimal_match_rewards': np.std(rewards_match_heuristic)}

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

save_path = get_save_path('matching',save_name,seed,use_date=save_with_date)

delete_duplicate_results('matching',save_name,data)

json.dump(data,open('../results/'+save_path,'w'))
