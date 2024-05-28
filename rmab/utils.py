import numpy as np
from datetime import datetime 
import glob 
import json 
import os 
import secrets
from scipy.stats import norm, beta
import scipy.stats as st
import math
import torch
import resource 


def get_stationary_distribution(P):
    """Given a Markov Chain, P, get its stationary distribution
    
    Arguments: 
        P: Square numpy array representing transition probabilities
    
    Returns: Vector of stationary probabilities"""

    eigenvalues, eigenvectors = np.linalg.eig(P.T)  # Transpose P to find left eigenvectors

    # Find the index of the eigenvalue equal to 1
    stationary_index = np.where(np.isclose(eigenvalues, 1))[0][0]

    # Get the corresponding left eigenvector
    stationary_distribution = np.real(eigenvectors[:, stationary_index])
    stationary_distribution /= np.sum(stationary_distribution)  # Normalize to ensure it sums to 1

    return stationary_distribution

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
        load_file = json.load(open(file_name,"r"))

        if 'parameters' in load_file and load_file['parameters'] == data['parameters']:
            try:
                os.remove(file_name)
            except OSError as e:
                print(f"Error deleting {file_name}: {e}")

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
        load_file = json.load(open(file_name,"r"))

        for p in parameters:
            if p not in load_file['parameters'] or load_file['parameters'][p] != parameters[p]:
                break 
        else:
            ret_results.append(load_file)
    return ret_results

class Memoizer:
    """ improve performance of memoizing solutions (to QP and WI value iteration) """
    def __init__(self, method):
        self.method = method
        self.solved_p_vals = {}

    def to_key(self, input1, input2):
        """ convert inputs to a key

        QP: inputs: LCB and UCB transition probabilities
        UCB and extreme: inputs - estimated transition probabilities and initial state s0 """
        if self.method in ['lcb_ucb', 'QP', 'QP-min']:
            lcb, ucb = input1, input2
            p_key = (np.round(lcb, 4).tobytes(), np.round(ucb, 4).tobytes())
        elif self.method in ['p_s', 'optimal', 'UCB', 'extreme', 'ucw_value']:
            transitions, state = input1, input2
            p_key = (np.round(transitions, 4).tobytes(), state)
        elif self.method in ['lcb_ucb_s_lamb']:
            lcb, ucb = input1
            s, lamb_val = input2
            p_key = (np.round(lcb, 4).tobytes(), np.round(ucb, 4).tobytes(), s, lamb_val)
        else:
            raise Exception(f'method {self.method} not implemented')

        return p_key

    def check_set(self, input1, input2):
        p_key = self.to_key(input1, input2)
        if p_key in self.solved_p_vals:
            return self.solved_p_vals[p_key]
        return -1

    def add_set(self, input1, input2, wi):
        p_key = self.to_key(input1, input2)
        self.solved_p_vals[p_key] = wi

def is_pareto_optimal(point, data):
    """Determine if a data point is pareto optimal
    
    Arguments:
        point: Numpy Array (x,y)
        data: List of Numpy Arrays of x,y
    
    Returns: Boolean; is the data point pareto optimal"""

    for other_point in data:
        if point[0] < other_point[0] and point[1] < other_point[1]:
            return False
    return True

def filter_pareto_optimal(data):
    """Reduce a list of numpy pairs to only the pareto optimal points
    
    Arguments:
        data: List of numpy arrays (x,y)
    
    Returns: List of numpy arrays (x,y)"""

    pareto_optimal_points = []
    for point in data:
        if is_pareto_optimal(point, data):
            pareto_optimal_points.append(point)
    return pareto_optimal_points

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


def custom_reward(s,a,match_probabilities,custom_reward_type,reward_parameters):
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
    elif custom_reward_type == "probability":
        probs = s*a*match_probabilities
        return 1-np.prod(1-probs)
    elif custom_reward_type == "two_by_two":
        probs = s*a*match_probabilities
        value_by_combo = {
            '0000': 0, 
            '1000': probs[0], 
            '0100': probs[1],
            '0010': probs[2],
            '0001': probs[3], 
            '1100': max(probs[0],probs[1]), 
            '1001': probs[0]+probs[3], 
            '1010': max(probs[0],probs[2]),
            '0110': max(probs[1],probs[2]), 
            '0101': probs[1]+probs[3],
            '0011': max(probs[2],probs[3]),
            '0111': max(probs[1]+probs[3],probs[2]),
            '1011': max(probs[0]+probs[3],probs[2]),
            '1101': max(probs[0]+probs[3],probs[1]+probs[3]),
            '1110': max(probs[0],max(probs[1],probs[2])),
            '1111': max(max(probs[0],probs[1])+probs[3],probs[2])
        }

        str_state_action = s*a 
        str_state_action = ''.join([str(i) for i in str_state_action])
        val = value_by_combo[str_state_action]

        return val 
    elif custom_reward_type == "linear":
        probs = s*a*match_probabilities
        return np.sum(probs)
    else:
        raise Exception("Reward type {} not found".format(custom_reward_type))  

def custom_reward_contextual(s,a,match_probabilities,custom_reward_type,reward_parameters,context):
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

    if custom_reward_type == "set_cover":
        num_elements = reward_parameters['universe_size']
        all_nums = set([i for i in range(0,num_elements)])
        all_seen = set() 
        for i in range(len(s)):
            if s[i]*a[i] == 1:
                all_seen = all_seen.union(match_probabilities[i])
        return len(all_nums.intersection(all_seen))
    elif custom_reward_type == "max":
        match_probabilities = match_probabilities + context 
        match_probabilities = np.max(match_probabilities,0)
        probs = s*a*match_probabilities
        return np.max(probs) 
    elif custom_reward_type == "probability":
        match_probabilities = match_probabilities + context 
        match_probabilities = np.clip(match_probabilities,0,1)
        probs = s*a*match_probabilities
        return 1-np.prod(1-probs)
    elif custom_reward_type == "two_by_two":
        probs = s*a*match_probabilities
        value_by_combo = {
            '0000': 0, 
            '1000': probs[0], 
            '0100': probs[1],
            '0010': probs[2],
            '0001': probs[3], 
            '1100': max(probs[0],probs[1]), 
            '1001': probs[0]+probs[3], 
            '1010': max(probs[0],probs[2]),
            '0110': max(probs[1],probs[2]), 
            '0101': probs[1]+probs[3],
            '0011': max(probs[2],probs[3]),
            '0111': max(probs[1]+probs[3],probs[2]),
            '1011': max(probs[0]+probs[3],probs[2]),
            '1101': max(probs[0]+probs[3],probs[1]+probs[3]),
            '1110': max(probs[0],max(probs[1],probs[2])),
            '1111': max(max(probs[0],probs[1])+probs[3],probs[2])
        }

        str_state_action = s*a 
        str_state_action = ''.join([str(i) for i in str_state_action])
        val = value_by_combo[str_state_action]

        return val 
    elif custom_reward_type == "linear":
        match_probabilities = match_probabilities + context 
        match_probabilities = np.clip(match_probabilities,0,1)
        probs = s*a*match_probabilities
        return np.sum(probs)
    else:
        raise Exception("Reward type {} not found".format(custom_reward_type))  


def generate_random_context(context_dim):
    """Generate a random context of a given context dimension
    
    Arguments:
        context_dim: Integer, number of contexts
        
    Returns: Vector of size context_dim"""

    return np.random.uniform(-1,1,(context_dim))

def get_average_context(context_dim):
    """Generate a random context of a given context dimension
    
    Arguments:
        context_dim: Integer, number of contexts
        
    Returns: Vector of size context_dim"""

    return np.zeros((context_dim))

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