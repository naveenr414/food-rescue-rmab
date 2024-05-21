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

# # Semi Synthetic Experiments

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

from rmab.simulator import RMABSimulator, run_heterogenous_policy, get_discounted_reward, create_random_transitions
from rmab.omniscient_policies import *
from rmab.dqn_policies import *
from rmab.fr_dynamics import get_all_transitions, get_db_data, get_all_transitions_partition
from rmab.mcts_policies import *
from rmab.utils import get_save_path, delete_duplicate_results, create_prob_distro
import resource

torch.cuda.set_per_process_memory_fraction(0.5)
torch.set_num_threads(1)
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 51
    n_arms      = 4
    volunteers_per_arm = 1
    budget      = 2
    discount    = 0.9
    alpha       = 3 
    n_episodes  = 105
    episode_len = 50 
    n_epochs    = 1
    save_with_date = False 
    TIME_PER_RUN = 0.01 * 1000
    lamb = 0.5
    prob_distro = 'uniform'
    reward_type = "linear"
    reward_parameters = {'universe_size': 20, 'arm_set_low': 0, 'arm_set_high': 1}
    policy_lr=5e-3
    value_lr=1e-4
    train_iterations = 30
    test_iterations = 30
    out_folder = 'iterative'
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
    parser.add_argument('--time_per_run',      '-t', help='time per MCTS run', type=float, default=.01*1000)
    parser.add_argument('--policy_lr', help='Learning Rate Policy', type=float, default=5e-3)
    parser.add_argument('--value_lr', help='Learning Rate Value', type=float, default=1e-4)
    parser.add_argument('--train_iterations', help='Number of MCTS train iterations', type=int, default=30)
    parser.add_argument('--test_iterations', help='Number of MCTS test iterations', type=int, default=30)
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
    TIME_PER_RUN = args.time_per_run
    prob_distro = args.prob_distro
    policy_lr = args.policy_lr 
    value_lr = args.value_lr 
    out_folder = args.out_folder
    train_iterations = args.train_iterations 
    test_iterations = args.test_iterations 
    reward_type = args.reward_type
    reward_parameters = {'universe_size': args.universe_size,
                        'arm_set_low': args.arm_set_low, 
                        'arm_set_high': args.arm_set_high}
    time_limit = args.time_limit 

save_name = secrets.token_hex(4)  
# -

n_states = 2
n_actions = 2

np.random.seed(seed)
all_population_size = 100 
max_transition_prob = 0.25
all_transitions = create_random_transitions(all_population_size,max_transition_prob)


def partition_volunteers(probs_by_num,num_by_section):
    total = sum([len(probs_by_num[i]) for i in probs_by_num])
    num_per_section = total//num_by_section

    nums_by_partition = []
    current_count = 0
    current_partition = []

    keys = sorted(probs_by_num.keys())

    for i in keys:
        if current_count >= num_per_section*(len(nums_by_partition)+1):
            nums_by_partition.append(current_partition)
            current_partition = []
        
        current_partition.append(i)
        current_count += len(probs_by_num[i])
    return nums_by_partition


if prob_distro == "food_rescue_top":
    all_population_size = 20 
    probs_by_user = json.load(open("../../results/food_rescue/match_probs.json","r"))
    donation_id_to_latlon, recipient_location_to_latlon, rescues_by_user, all_rescue_data, user_id_to_latlon = get_db_data()
    probs_by_num = {}
    for i in rescues_by_user:
        if str(i) in probs_by_user and probs_by_user[str(i)] > 0 and len(rescues_by_user[i]) >= 100:
            if len(rescues_by_user[i]) not in probs_by_num:
                probs_by_num[len(rescues_by_user[i])] = []
            probs_by_num[len(rescues_by_user[i])].append(probs_by_user[str(i)])

    partitions = partition_volunteers(probs_by_num,all_population_size)
    probs_by_partition = []

    for i in range(len(partitions)):
        temp_probs = []
        for j in partitions[i]:
            temp_probs += (probs_by_num[j])
        probs_by_partition.append(temp_probs)

    all_transitions = get_all_transitions_partition(all_population_size,partitions)

    for i,partition in enumerate(partitions):
        current_transitions = np.array(all_transitions[i])
        partition_scale = np.array([len(probs_by_num[j]) for j in partition])
        partition_scale = partition_scale/np.sum(partition_scale)
        prod = current_transitions*partition_scale[:,np.newaxis,np.newaxis,np.newaxis]
        new_transition = np.sum(prod,axis=0)
        all_transitions[i] = new_transition
    all_transitions = np.array(all_transitions)

if prob_distro == "food_rescue":
    all_population_size = 100 

    probs_by_user = json.load(open("../../results/food_rescue/match_probs.json","r"))
    donation_id_to_latlon, recipient_location_to_latlon, rescues_by_user, all_rescue_data, user_id_to_latlon = get_db_data()
    probs_by_num = {}
    for i in rescues_by_user:
        if str(i) in probs_by_user and probs_by_user[str(i)] > 0 and len(rescues_by_user[i]) >= 3:
            if len(rescues_by_user[i]) not in probs_by_num:
                probs_by_num[len(rescues_by_user[i])] = []
            probs_by_num[len(rescues_by_user[i])].append(probs_by_user[str(i)])

    partitions = partition_volunteers(probs_by_num,all_population_size)
    probs_by_partition = []
    all_transitions = get_all_transitions_partition(all_population_size,partitions)

    for i in range(len(partitions)):
        temp_probs = []
        for j in partitions[i]:
            temp_probs += (probs_by_num[j])
        probs_by_partition.append(temp_probs)

    for i,partition in enumerate(partitions):
        current_transitions = np.array(all_transitions[i])
        partition_scale = np.array([len(probs_by_num[j]) for j in partition])
        partition_scale = partition_scale/np.sum(partition_scale)
        prod = current_transitions*partition_scale[:,np.newaxis,np.newaxis,np.newaxis]
        new_transition = np.sum(prod,axis=0)
        all_transitions[i] = new_transition
    all_transitions = np.array(all_transitions)

if prob_distro == "high_prob":
    np.random.seed(seed)
    all_population_size = 100 
    max_transition_prob = 1.0
    all_transitions = create_random_transitions(all_population_size,max_transition_prob)

if prob_distro == "one_time":
    np.random.seed(seed)
    all_population_size = 100 
    max_transition_prob = 1.0
    all_transitions = np.zeros((all_population_size,2,2,2))
    all_transitions[:,:,1,0] = 1
    all_transitions[:,1,0,1] = 1
    all_transitions[:,0,0,0] = 1


def create_environment(seed):
    random.seed(seed)
    np.random.seed(seed)

    all_features = np.arange(all_population_size)
    N = all_population_size*volunteers_per_arm
    if reward_type == "set_cover":
        if prob_distro == "fixed":
            match_probabilities = []
            set_sizes = [int(reward_parameters['arm_set_low']) for i in range(N)]
            for i in range(N):
                s = set() 
                while len(s) < set_sizes[i]:
                    s.add(np.random.randint(0,reward_parameters['universe_size']))
                match_probabilities.append(s)
        else:
            set_sizes = [np.random.randint(int(reward_parameters['arm_set_low']),int(reward_parameters['arm_set_high'])+1) for i in range(N)]
            match_probabilities = [] 
            
            for i in range(N):
                temp_set = set() 
                
                while len(temp_set) < set_sizes[i]:
                    temp_set.add(np.random.randint(0,reward_parameters['universe_size']))
                match_probabilities.append(temp_set)
    elif prob_distro == "food_rescue" or prob_distro == "food_rescue_top":
        match_probabilities = [np.random.choice(probs_by_partition[i//volunteers_per_arm]) for i in range(N)] 
    else:
        match_probabilities = [np.random.uniform(reward_parameters['arm_set_low'],reward_parameters['arm_set_high']) for i in range(N)]

    simulator = RMABSimulator(all_population_size, all_features, all_transitions,
                n_arms, volunteers_per_arm, episode_len, n_epochs, n_episodes, budget, discount,number_states=n_states, reward_style='custom',match_probability_list=match_probabilities,TIME_PER_RUN=TIME_PER_RUN)
    simulator.reward_type = reward_type 
    simulator.reward_parameters = reward_parameters 
    return simulator 


def run_multi_seed(seed_list,policy,is_mcts=False,per_epoch_function=None,train_iterations=0,test_iterations=0,test_length=20,avg_reward=0,num_samples=100):
    memories = []
    scores = {
        'reward': [],
        'time': [], 
        'match': [], 
        'active_rate': [],
    }

    for seed in seed_list:
        simulator = create_environment(seed)
        simulator.time_limit = time_limit
        simulator.avg_reward = avg_reward
        simulator.num_samples = num_samples
        simulator.mcts_train_iterations = train_iterations
        simulator.mcts_test_iterations = 400
        simulator.policy_lr = policy_lr
        simulator.value_lr = value_lr
        simulator.mcts_depth = 2
        simulator.shapley_iterations = 1000 

        if prob_distro == "one_time":
            N = n_arms*volunteers_per_arm
            simulator.first_init_states = np.array([[[1 for i in range(N)] for i in range(n_episodes)]])
            random.seed(seed)
            shuffled_list = [reward_parameters['arm_set_high'] for i in range(2)] + [reward_parameters['arm_set_high'] for i in range(N-2)]
            random.shuffle(shuffled_list)

            simulator.match_probability_list[simulator.cohort_selection[0]] = shuffled_list

        if is_mcts:
            match, active_rate, memory = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb,should_train=True,test_T=test_length,get_memory=True,per_epoch_function=per_epoch_function)
        else:
            match, active_rate = run_heterogenous_policy(simulator, n_episodes, n_epochs, discount,policy,seed,lamb=lamb,should_train=False,test_T=test_length,per_epoch_function=per_epoch_function)
        num_timesteps = match.size
        match = match.reshape((num_timesteps//episode_len,episode_len))
        active_rate = active_rate.reshape((num_timesteps//episode_len,episode_len))

 
        time_whittle = simulator.time_taken
        discounted_reward = get_discounted_reward(match,active_rate,discount,lamb)
        scores['reward'].append(discounted_reward)
        scores['time'].append(time_whittle)
        scores['match'].append(np.mean(match))
        scores['active_rate'].append(np.mean(active_rate))
        if is_mcts:
            memories.append(memory)

    return scores, memories, simulator

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
        'time_per_run': TIME_PER_RUN, 
        'prob_distro': prob_distro, 
        'policy_lr': policy_lr, 
        'value_lr': value_lr, 
        'reward_type': reward_type, 
        'universe_size': reward_parameters['universe_size'],
        'arm_set_low': reward_parameters['arm_set_low'], 
        'arm_set_high': reward_parameters['arm_set_high'],
        'time_limit': time_limit
        } 

# ## Index Policies

seed_list = [seed]

# +
policy = whittle_policy
name = "linear_whittle"

rewards, memory, simulator = run_multi_seed(seed_list,policy,test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = mcts_policy
name = "mcts"

rewards, memory, simulator = run_multi_seed(seed_list,policy,test_length=episode_len*(n_episodes%50),test_iterations=400)
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = greedy_policy
name = "greedy"

rewards, memory, simulator = run_multi_seed(seed_list,policy,test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = random_policy
name = "random"

rewards, memory, simulator = run_multi_seed(seed_list,policy,test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
policy = whittle_activity_policy
name = "whittle_activity"

rewards, memory, simulator = run_multi_seed(seed_list,policy,test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))
# -

if n_arms*volunteers_per_arm <= 10:
    policy = dqn_policy_greedy
    name = "dqn"

    print("Running DQN")

    rewards, memory, simulator = run_multi_seed(seed_list,policy,is_mcts=True,avg_reward=np.mean(results['linear_whittle_reward'][0]),test_length=episode_len*(n_episodes%50))
    results['{}_reward'.format(name)] = rewards['reward']
    results['{}_match'.format(name)] =  rewards['match'] 
    results['{}_active'.format(name)] = rewards['active_rate']
    results['{}_time'.format(name)] =  rewards['time']
    print(np.mean(rewards['reward']))

# +
policy = dqn_with_steps
name = "dqn_step"

print("Running DQN Step")

rewards, memory, simulator = run_multi_seed(seed_list,policy,is_mcts=True,avg_reward=np.mean(results['linear_whittle_reward'][0]),test_length=episode_len*(n_episodes%50))
results['{}_reward'.format(name)] = rewards['reward']
results['{}_match'.format(name)] =  rewards['match'] 
results['{}_active'.format(name)] = rewards['active_rate']
results['{}_time'.format(name)] =  rewards['time']
print(np.mean(rewards['reward']))

# +
# policy = dqn_with_stablization_steps
# name = "dqn_stable_step"

# print("Running DQN Step")

# rewards, memory, simulator = run_multi_seed(seed_list,policy,is_mcts=True,avg_reward=np.mean(results['linear_whittle_reward'][0]),test_length=episode_len,num_samples=1024)
# results['{}_reward'.format(name)] = rewards['reward']
# results['{}_match'.format(name)] =  rewards['match'] 
# results['{}_active'.format(name)] = rewards['active_rate']
# results['{}_time'.format(name)] =  rewards['time']
# print(np.mean(rewards['reward']))

# +
# if n_arms * volunteers_per_arm <= 4:
#     policy = q_iteration_policy
#     per_epoch_function = q_iteration_custom_epoch()
#     name = "optimal"

#     rewards, memory, simulator = run_multi_seed(seed_list,policy,per_epoch_function=per_epoch_function,test_length=episode_len*(n_episodes%50))
#     results['{}_reward'.format(name)] = rewards['reward']
#     results['{}_match'.format(name)] =  rewards['match'] 
#     results['{}_active'.format(name)] = rewards['active_rate']
#     results['{}_time'.format(name)] =  rewards['time']
#     print(np.mean(rewards['reward']))
# -

# ## Write Data

save_path = get_save_path(out_folder,save_name,seed,use_date=save_with_date)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))


