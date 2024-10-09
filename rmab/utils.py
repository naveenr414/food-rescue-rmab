import numpy as np
from datetime import datetime 
import glob 
import ujson as json 
import os 
import secrets
from scipy.stats import norm, beta
import scipy.stats as st
import math
import torch
import resource 
import time
from copy import deepcopy
import scipy 
import random 


def get_save_path(folder_name,result_name,seed,use_date=False):
    """Create a string, file_name, which is the name of the file to save
    
    Arguments:
        project_name: String, such as 'baseline_bandit'
        seed: Integer, such as 43
        
    Returns: String, such as 'baseline_bandit_43_2023-05-06'"""

    seedstr = str(seed) 
    suffix = "{}/{}_{}".format(folder_name,result_name, seedstr)

    current_datetime = datetime.now()
    nice_datetime_string = current_datetime.strftime("%B_%d_%Y_%I:%M_%p").replace(":","_")
    if use_date:
        suffix += "_"+nice_datetime_string+"_"+secrets.token_hex(2)

    suffix += '.json'
    return suffix

def delete_duplicate_results(folder_name,result_name,data):
    """Delete all results with the same parameters, so it's updated
    
    Arguments:
        folder_name: Name of the results folder to look in
        results_name: What experiment are we running (hyperparameter for e.g.)
        data: Dictionary, with the key parameters
        
    Returns: Nothing
    
    Side Effects: Deletes .json files from folder_name/result_name..."""

    all_results = glob.glob("../../results/{}/{}*.json".format(folder_name,result_name))

    for file_name in all_results:
        try:
            f = open(file_name)
            first_few = f.read(1000)
            first_few = first_few.split("}")[0]+"}}"
            load_file = json.loads(first_few)['parameters']
            if load_file == data['parameters']:
                try:
                    os.remove(file_name)
                except OSError as e:
                    print(f"Error deleting {file_name}: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

def get_results_matching_parameters(folder_name,result_name,parameters):
    """Get a list of dictionaries, with data, which match some set of parameters
    
    Arguments:
        folder_name: String, which folder the data is located in
        result_name: String, what the suffix is for the dataset
        parameters: Dictionary with key,values representing some known parameters
        
    Returns: List of Dictionaries"""

    all_results = glob.glob("../../results/{}/{}*.json".format(folder_name,result_name))
    ret_results = []

    for file_name in all_results:
        f = open(file_name)
        first_few = f.read(1000)
        first_few = first_few.split("}")[0]+"}}"
        load_file = json.loads(first_few)
        for p in parameters:
            if p not in load_file['parameters'] or load_file['parameters'][p] != parameters[p]:
                break 
        else:
            load_file = json.load(open(file_name,"r"))
            ret_results.append(load_file)
    return ret_results

def binary_to_decimal(binary_list):
    """Turn 0-1 lists into a number, for state representation
    
    Arguments:
        binary_list: List of 0,1
    
    Returns: Integer base-10 represnetation"""

    decimal_value = 0
    for bit in binary_list:
        decimal_value = decimal_value * 2 + bit
    return decimal_value

def list_to_binary(a,n_arms):
    """Given a list of the form [0,3,5], return a binary
        array of length n_arms with 1 if i is in a, 0 otherwise
        For example, [1,0,0,1,0,1]
    
    Arguments: a, numpy array or list
        n_arms: Integer, length of the return list
    
    Returns: 0-1 List of length n_arms"""

    return np.array([1 if i in a else 0 for i in range(n_arms)])

def create_prob_distro(prob_distro,N):
    """Create match probabilities for N volunteers according to some distro.
    
    Arguments:
        prob_distro: String, 'uniform', 'uniform_small', 'uniform_large', 
            or 'normal'
        N: Number of total volunteers
        
    Returns: List of floats"""

    if prob_distro == 'uniform':
        match_probabilities = [np.random.random() for i in range(N)] 
    elif prob_distro == 'uniform_small':
        match_probabilities = [np.random.random()/4 for i in range(N)] 
    elif prob_distro == 'uniform_large':
        match_probabilities = [np.random.random()/4+0.75 for i in range(N)] 
    elif prob_distro == 'normal':
        match_probabilities = [np.clip(np.random.normal(0.25, 0.1),0,1) for i in range(N)] 
    else:
        raise Exception("{} probability distro not found".format(prob_distro))

    return match_probabilities

def haversine(lat1, lon1, lat2, lon2):
    """Compute the distance, in miles, between two lat-lon coordinates

    Arguments:
        lat1: Float, 1st lattitude
        lon1: Float, 1st longitude
        lat2: Float, 2nd lattitude
        lon2: Float, 2nd longitude
    
    Returns: Float, distnace in miles"""
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    r = 3956
    return r * c

def binary_search_count(arr, element):
    """
    Performs binary search on a sorted list and returns the number of elements less than the given element.

    Parameters:
        arr (list): Sorted list of elements.
        element: Element to find the count of elements less than it.

    Returns:
        int: Number of elements less than the given element.
    """
    left = 0
    right = len(arr) - 1
    count = 0

    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] < element:
            count = mid + 1  # Increment count for current position
            left = mid + 1
        else:
            right = mid - 1

    return count

def one_hot(index,length):
    """Create a list with length-1 0s, and a 1 at location index
        e.g. one_hot(3,5) = [0,0,0,1,0]

    Arguments:
        index: Integer, number, where the 1 is
        length: Integer, number, length of the lsit
    
    Returns: List with length-1 0s and one 1 """
    s = [0 for i in range(length)]
    s[index] = 1
    return np.array(s)

def one_hot_fixed(index,length,fixed):
    """Create a list with length-1 0s, and a 1 at location index, and at 
        some indices, fixed 
        e.g. one_hot(3,5,[1,2]) = [0,1,1,1,0]

    Arguments:
        index: Integer, number, where the 1 is
        length: Integer, number, length of the list 
        fixed: List of integers, which we set to 1
    
    Returns: List with length-1 0s and one 1 """
    s = np.array([0 for i in range(length)])
    s[index] = 1
    s[fixed] = 1
    return s


def custom_reward(s,a,match_probabilities,custom_reward_type,reward_parameters,active_states,state_as_num=None):
    """Custom defined submodular reward which is maximized by
        each policy
    
    Arguments:
        s: Numpy array for the state of length N
        a: Numpy array for the action of lenghth N
        match_probabilities: Numpy array with information for each arm
            Of length N
            For example, for set cover, match_probabilities contains
            The set corresponding to each arm
    
    Returns: Float, reward"""

    if state_as_num is None:
        state_as_num = np.array([1 if s[i] in active_states else 0 for i in range(len(s))])

    if custom_reward_type == "set_cover":
        num_elements = reward_parameters['universe_size']
        all_nums = set([i for i in range(0,num_elements)])
        all_seen = set() 
        for i in range(len(s)):
            if s[i]*a[i] == 1:
                all_seen = all_seen.union(match_probabilities[i])
        return len(all_nums.intersection(all_seen))
    elif custom_reward_type == "max":
        probs = s*a*match_probabilities
        return np.max(probs) 
    elif custom_reward_type == "std":
        probs = s*a*match_probabilities
        val_probs = [probs[i] for i in range(len(probs)) if a[i] == s[i] == 1]
        if len(val_probs) == 0:
            return 0
        return np.std(val_probs) 
    elif custom_reward_type == "min":
        probs = s*a*match_probabilities
        val_probs = [i for i in probs if i>0]
        if len(val_probs) == 0:
            return 0
        else:
            return np.min(val_probs) 
    elif custom_reward_type == "probability":
        probs = state_as_num * a * match_probabilities
        return 1 - np.prod(1 - probs)
    elif custom_reward_type == "probability_context":
        probs = s*a*match_probabilities
        return 1-np.prod(1-probs)
    elif custom_reward_type == "linear":
        real_s = np.array([1 if s[i] in active_states else 0 for i in range(len(s))])
        probs = real_s*a*match_probabilities
        return np.sum(probs)
    else:
        raise Exception("Reward type {} not found".format(custom_reward_type))  

def contextual_custom_reward(s,a,match_probabilities,custom_reward_type,reward_parameters,active_states,context,state_as_num=None):
    """Custom defined submodular reward which is maximized by
        each policy
    
    Arguments:
        s: Numpy array for the state of length N
        a: Numpy array for the action of lenghth N
        match_probabilities: Numpy array with information for each arm
            Of length N
            For example, for set cover, match_probabilities contains
            The set corresponding to each arm
    
    Returns: Float, reward"""
    if state_as_num is None:
        state_as_num = np.array([1 if s[i] in active_states else 0 for i in range(len(s))])

    if custom_reward_type == "probability_context":
        new_match_probabilities = context
        probs = state_as_num*a*new_match_probabilities
        return 1-np.prod(1-probs)
    else:
        return custom_reward(s,a,match_probabilities,custom_reward_type,reward_parameters,active_states,state_as_num=state_as_num)


def partition_volunteers(probs_by_num,num_by_section):
    """Given a list of volunteer probabilities, partition this
        for each arm 
        E.g., partition_volunteers([[0.1,0.2],[0.3,0.4],[0.5]],[[0,1],[2]])
            Returns: [[0.1,0.2,0.3,0.4],[0.5]]

    Arguments:
        probs_by_num: List of lists of floats, corresponding to match probabilities
            for each volunteer who completed X trips
        num_by_section: List of list of integers; which trip numbers correspond to  
            which partitions
    
    Returns: List of list of floats, match probabilities per arm
        The values m_{i} are chosen from this
    """

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

def restrict_resources():
    """Set the system to only use a fraction of the memory/CPU/GPU available
    
    Arguments: None
    
    Returns: None
    
    Side Effects: Makes sure that a) Only 50% of GPU is used, b) 1 Thread is used, and c) 30 GB of Memory"""

    torch.cuda.set_per_process_memory_fraction(0.5)
    torch.set_num_threads(1)
    resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 * 1024 * 1024, -1))

def shapley_index_custom(env,state,memoizer_shapley = {},idx=-1):
    """Compute the Shapley index for matching; how much
        does match probability increase when using some particular arm
        u_{i}(s_{i})
        
    Arguments:
        env: RMAB Simulator environment
        state: Numpy array of 0-1 states for each volunteer, s_{i}
        memoizer_shapley: Dictionary, to store previously computed Shapley indices
        
    Returns: Two things, shapley index, and updated dictionary"""

    shapley_indices = [0 for i in range(len(state))]
    state_str = " ".join([str(i) for i in state])
    if state_str in memoizer_shapley:
        return memoizer_shapley[state_str], memoizer_shapley
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    corresponding_probabilities = match_probabilities
    num_random_combos = env.shapley_iterations 

    combinations = np.zeros((num_random_combos, len(corresponding_probabilities)), dtype=int)
    budget = env.budget 
    if len(corresponding_probabilities) <= env.budget-1:
        if len(corresponding_probabilities) == 1:
            return match_probabilities * state, memoizer_shapley
        else: 
            budget = 2
    budget_probs = np.array([scipy.special.comb(len(corresponding_probabilities),k) for k in range(0,budget)])
    budget_probs /= np.sum(budget_probs)

    for i in range(num_random_combos):
        k = random.choices(list(range(len(budget_probs))), weights=budget_probs,k=1)[0]
        ones_indices = random.sample(list(range(len(corresponding_probabilities))),k)
        combinations[i, ones_indices] = 1

    state = [int(i) for i in state]
    scores = []
    for i in range(num_random_combos):
        combo = combinations[i]
        scores.append(custom_reward(state,combo,corresponding_probabilities,env.reward_type,env.reward_parameters,env.active_states))

    scores = np.array(scores)

    num_by_shapley_index = np.zeros(len(state))
    for j,combo in enumerate(combinations):
        action = deepcopy(combo) 

        if idx != -1:
            i = idx 
            if combo[i] == 0:
                action[i] = 1
                shapley_indices[i] += custom_reward(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters,env.active_states) - scores[j]
                num_by_shapley_index[i] += 1
                action[i] = 0
        else:
            for i in range(len(state)):
                if combo[i] == 0:
                    action[i] = 1
                    shapley_indices[i] += custom_reward(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters,env.active_states) - scores[j]
                    num_by_shapley_index[i] += 1
                    action[i] = 0
    
    if idx != -1:
        return shapley_indices[i]/num_by_shapley_index[i]
    shapley_indices /= num_by_shapley_index
    memoizer_shapley[state_str] = shapley_indices

    return shapley_indices, memoizer_shapley

def shapley_index_custom_contexts(env,state,context,memoizer_shapley = {},idx=-1):
    """Compute the Shapley index for matching; how much
        does match probability increase when using some particular arm
        u_{i}(s_{i})
        
    Arguments:
        env: RMAB Simulator environment
        state: Numpy array of 0-1 states for each volunteer, s_{i}
        memoizer_shapley: Dictionary, to store previously computed Shapley indices
        
    Returns: Two things, shapley index, and updated dictionary"""
    state_as_num = np.array([int(i in env.active_states) for i in state])

    shapley_indices = [0 for i in range(len(state))]
    state_str = " ".join([str(i) for i in state])
    if "context" in env.reward_type:
        state_str+=str(context)
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    corresponding_probabilities = match_probabilities
    num_random_combos = env.shapley_iterations 

    combinations = np.zeros((num_random_combos, len(corresponding_probabilities)), dtype=int)
    budget = env.budget 
    if len(corresponding_probabilities) <= env.budget-1:
        if len(corresponding_probabilities) == 1:
            return match_probabilities * state, memoizer_shapley
        else: 
            budget = 2
    budget_probs = np.array([scipy.special.comb(len(corresponding_probabilities),k) for k in range(0,budget)])
    budget_probs /= np.sum(budget_probs)

    chosen_indices = random.choices(list(range(len(budget_probs))), weights=budget_probs, k=num_random_combos)
    # Then, generate random samples for ones_indices
    for i in range(num_random_combos):
        k = chosen_indices[i]
        ones_indices = random.sample(list(range(len(corresponding_probabilities))), k)
        combinations[i, ones_indices] = 1
    state = [int(i) for i in state]
    scores = np.zeros(num_random_combos)

    if env.reward_type == "probability_context":
        scores = 1-np.prod(1-state_as_num*combinations*corresponding_probabilities,axis=1)
    else:
        for i in range(num_random_combos):
            if idx != -1 and combinations[i,idx] == 1:
                continue  
            scores[i] = contextual_custom_reward(state,combinations[i],corresponding_probabilities,env.reward_type,env.reward_parameters,env.active_states,context,state_as_num=state_as_num)
        scores = np.array(scores)

    num_by_shapley_index = np.zeros(len(state))

    if idx!=-1 and env.reward_type == "probability_context":
        num_by_shapley_index[idx] = np.sum(combinations[:,idx])
        combinations[:,idx] = 1
        new_scores = 1-np.prod(1-state_as_num*combinations*corresponding_probabilities,axis=1)
        shapley_indices[idx] = np.sum(new_scores)-np.sum(scores)
        return shapley_indices[idx]/num_by_shapley_index[idx]
    else:
        for j,combo in enumerate(combinations):
            action = combo

            if idx != -1:
                i = idx 
                if combo[i] == 0:
                    action[i] = 1
                    new_reward = contextual_custom_reward(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters,env.active_states,context,state_as_num=state_as_num)
                    shapley_indices[i] += new_reward - scores[j]
                    num_by_shapley_index[i] += 1
                    action[i] = 0
            else:
                for i in range(len(state)):
                    if combo[i] == 0:
                        action[i] = 1
                        shapley_indices[i] += contextual_custom_reward(state,np.array(action),corresponding_probabilities,env.reward_type,env.reward_parameters,env.active_states,context,state_as_num=state_as_num) - scores[j]
                        num_by_shapley_index[i] += 1
                        action[i] = 0
    if idx != -1:
        return shapley_indices[i]/num_by_shapley_index[i]
    shapley_indices /= num_by_shapley_index
    memoizer_shapley[state_str] = shapley_indices

    return shapley_indices, memoizer_shapley



def shapley_index_custom_fixed(env,state,memoizer_shapley,arms_pulled,context):
    """Compute the Shapley index for matching; how much
        does match probability increase when using some particular arm
        Assume that some arms were already pulled 
        
    Arguments:
        env: RMAB Simulator environment
        state: Numpy array of 0-1 states for each volunteer
        memoizer_shapley: Dictionary, to store previously computed Shapley indices
        arms_pulled: Which arms were already pulled 
        
    Returns: Two things, shapley index, and updated dictionary"""

    shapley_indices = [0 for i in range(len(state))]
    state_str = " ".join([str(i) for i in state])
    if "context" in env.reward_type:
        state_str+=str(context)

    if state_str in memoizer_shapley:
        return memoizer_shapley[state_str], memoizer_shapley

    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    corresponding_probabilities = match_probabilities
    num_random_combos = env.shapley_iterations

    combinations = np.zeros((num_random_combos, len(corresponding_probabilities)), dtype=int)

    budget = env.budget-len(arms_pulled)

    # Fix for when the number of combinations is small (with respect to the budget)
    # In that scenario, we can essentially just manually compute
    if len(corresponding_probabilities) <= env.budget-1:
        if len(corresponding_probabilities) == 1:
            return match_probabilities * state, memoizer_shapley
        else: 
            budget = 2

    budget_probs = np.array([scipy.special.comb(len(corresponding_probabilities),k) for k in range(0,budget)])
    budget_probs /= np.sum(budget_probs)

    set_arms_pulled = set(arms_pulled)
    arms_not_pulled = [i for i in range(len(state)) if i not in set_arms_pulled]

    for i in range(num_random_combos):
        k = random.choices(list(range(len(budget_probs))), weights=budget_probs,k=1)[0]
        ones_indices = random.sample(arms_not_pulled,k)
        combinations[i, ones_indices] = 1
        combinations[i,arms_pulled] = 1

    state = [int(i) for i in state]
    state_as_num = np.array([int(i in env.active_states) for i in state])

    num_by_shapley_index = np.zeros(len(state))
    if env.reward_type == "probability_context":
        scores = 1-np.prod(1-state_as_num*combinations*corresponding_probabilities,axis=1)
    else:
        scores = []
        for i in range(num_random_combos):
            combo = combinations[i]
            scores.append(contextual_custom_reward(state,combo,corresponding_probabilities,env.reward_type,env.reward_parameters,env.active_states,context))

    scores = np.array(scores)
    idx = 0
    for j, combo in enumerate(combinations):
        action = np.array(combo)  # Avoid deepcopy, make a direct copy (lightweight)
        for i in range(len(state)):
            if combo[i] == 0:
                action[i] = 1
                idx += 1

                reward_diff = contextual_custom_reward(state, action, corresponding_probabilities, 
                                                    env.reward_type, env.reward_parameters, 
                                                    env.active_states, context,state_as_num=state_as_num) - scores[j]
                shapley_indices[i] += reward_diff
                num_by_shapley_index[i] += 1
                action[i] = 0  # Toggle back to 0

    for i in range(len(shapley_indices)):
        if num_by_shapley_index[i] > 0:
            shapley_indices[i] /= num_by_shapley_index[i] 
        else:
            shapley_indices[i] = 0
    memoizer_shapley[state_str] = shapley_indices

    return np.array(shapley_indices), memoizer_shapley

def compute_u_matrix(env,N,n_states):
    """Compute a matrices of values of u_{i}(s) for all i, s
    Useful to compute Shapley-Whittle indices
    
    Arguments:
        env: RMABSimulator Environment
        N: Integer, number of agents
        n_states: Number of total states
    
    Returns: numpy matrix of size N x number of states"""

    u_matrix = np.zeros((N,n_states))

    for s in range(n_states):
        for i in range(N):
            curr_state = [env.best_state for _ in range(N)]
            curr_state[i] = s
            u_matrix[i,s] = shapley_index_custom(env,curr_state,{},idx=i)
    return u_matrix