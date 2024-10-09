import random 
import numpy as np
from rmab.simulator import run_multi_seed

def random_policy(env,state,budget,lamb,memory, per_epoch_results):
    """Random policy that randomly notifies budget arms
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""


    N = len(state)
    selected_idx = random.sample(list(range(N)), budget)
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, None

seed_list=[43]
policy=random_policy
episode_len=50
n_episodes=5
parameters={'seed': 43,
  'n_arms': 10,
  'volunteers_per_arm': 1,
  'budget': 5,
  'discount': 0.9,
  'alpha': 3,
  'n_episodes': n_episodes,
  'episode_len': episode_len,
  'n_epochs': 1,
  'lamb': 0,
  'prob_distro': 'uniform',
  'reward_type': 'probability',
  'universe_size': 20,
  'arm_set_low': 0,
  'arm_set_high': 1,
  'time_limit': 100,
  'recovery_rate': 0}
rewards, memory, simulator = run_multi_seed(seed_list,policy,parameters,test_length=episode_len*n_episodes)
print(np.mean(rewards['reward']))
