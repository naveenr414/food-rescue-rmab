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
from rmab.baselines import optimal_whittle,  optimal_q_iteration, optimal_whittle_sufficient, optimal_neural_q_iteration
from rmab.fr_dynamics import get_all_transitions
from rmab.utils import get_save_path, delete_duplicate_results


is_jupyter = 'ipykernel' in sys.modules

if is_jupyter: 
    seed        = 42
    n_arms      = 5
    budget      = 3 
    discount    = 0.9
    alpha       = 3 
    n_episodes  = 30
    episode_len = 20 
    n_epochs    = 10
    save_name = 'combined_arms_{}'.format(n_arms)
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

lamb = 1

import logging
logging.disable(logging.CRITICAL)

if is_jupyter:
     np.random.seed(seed)
     random.seed(seed)
     register(
          id="Custom_Env/IdentityEnv", # Fill in the name of the custom environment, it can be freely modified
          entry_point="rmab.simulator:RMABSimulatorOpenRL", # Fill in the filename and class name of the custom environment
     )
     simulator_rl = make(id="Custom_Env/IdentityEnv", agent_num=10,
     all_population=all_population_size, all_features=all_features, all_transitions=all_transitions,
               cohort_size=n_arms, episode_len=episode_len, n_instances=n_epochs, n_episodes=n_episodes, budget=budget, number_states=n_states, reward_style='combined',lamb=lamb,match_probability=match_prob)
     simulator_rl.episode_len = episode_len
     neural_reward, neural_active_rate = optimal_neural_q_iteration(simulator_rl, budget,match_prob, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
     print(np.mean(neural_reward) + lamb*n_arms*neural_active_rate)

if is_jupyter:
    np.random.seed(seed)
    random.seed(seed)
    joint_combined_reward = optimal_q_iteration(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    joint_combined_active_rate = simulator.total_active/(joint_combined_reward.size*n_arms)
    print(np.mean(joint_combined_reward) + lamb*n_arms*joint_combined_active_rate)

if is_jupyter:
    np.random.seed(seed)
    random.seed(seed)
    sufficient_reward = optimal_whittle_sufficient(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    sufficient_active_rate = simulator.total_active/(sufficient_reward.size*n_arms)
    print(np.mean(sufficient_reward)+lamb*n_arms*sufficient_active_rate)

if is_jupyter:
    np.random.seed(seed)
    random.seed(seed)
    approximate_combined_reward = optimal_whittle(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    approximate_combined_active_rate = simulator.total_active/(approximate_combined_reward.size*n_arms)
    print(np.mean(approximate_combined_reward) + lamb*n_arms*approximate_combined_active_rate)

lamb_list = [0,1,2,4,8,16,32,64] 
lamb_list = [i/n_arms for i in lamb_list]


# +
approximate_match = []
approximate_active = []

sufficient_match = []
sufficient_active = []

neural_match = []
neural_active = []
# -

for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    approximate_combined_reward = optimal_whittle(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    approximate_combined_active_rate = simulator.total_active/(approximate_combined_reward.size*n_arms)

    approximate_match.append(np.mean(approximate_combined_reward))
    approximate_active.append(approximate_combined_active_rate)

for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    sufficient_reward = optimal_whittle_sufficient(simulator, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)
    sufficient_active_rate = simulator.total_active/(sufficient_reward.size*n_arms)

    sufficient_match.append(np.mean(sufficient_reward))
    sufficient_active.append(sufficient_active_rate)

for lamb in lamb_list:
    np.random.seed(seed)
    random.seed(seed)
    register(
        id="Custom_Env/IdentityEnv", # Fill in the name of the custom environment, it can be freely modified
        entry_point="rmab.simulator:RMABSimulatorOpenRL", # Fill in the filename and class name of the custom environment
    )
    simulator_rl = make(id="Custom_Env/IdentityEnv", agent_num=n_arms,
    all_population=all_population_size, all_features=all_features, all_transitions=all_transitions,
            cohort_size=n_arms, episode_len=episode_len, n_instances=n_epochs, n_episodes=n_episodes, budget=budget, number_states=n_states, reward_style='combined',lamb=lamb,match_probability=match_prob)
    simulator_rl.episode_len = episode_len
    neural_reward, neural_active_rate = optimal_neural_q_iteration(simulator_rl, budget,match_prob, n_episodes, n_epochs, discount,reward_function='combined',lamb=lamb)

    neural_match.append(np.mean(neural_reward))
    neural_active.append(neural_active_rate)

data = {
    'whittle_match': approximate_match, 
    'whittle_active': approximate_active,
    'sufficient_match': sufficient_match, 
    'sufficient_active': sufficient_active,
    'neural_match': neural_match, 
    'neural_active': neural_active, 
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

if is_jupyter:
    plt.plot(joint_match,joint_active,label='Q Iteration')
    plt.plot(sufficient_match,sufficient_active,label='Sufficient')
    plt.plot(approximate_match,approximate_active,label='Whittle')
    # plt.plot(neural_match,neural_active,label='PPO')
    plt.legend()
    plt.show()

save_path = get_save_path('combined',save_name,seed,use_date=save_with_date)

delete_duplicate_results('combined',save_name,data)

json.dump(data,open('../results/'+save_path,'w'))
