import numpy as np
from datetime import datetime 
import glob 
import json 
import os 
import secrets
from scipy.stats import norm

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

    all_results = glob.glob("../results/{}/{}*.json".format(folder_name,result_name))

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

    all_results = glob.glob("../results/{}/{}*.json".format(folder_name,result_name))
    ret_results = []

    for file_name in all_results:
        load_file = json.load(open(file_name,"r"))

        for p in parameters:
            if load_file['parameters'][p] != parameters[p]:
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

def get_valid_lcb_ucb(arm_p_lcb, arm_p_ucb):
    n_states, n_actions = arm_p_lcb.shape

    # enforce validity constraints
    assert n_actions == 2  # these checks only valid for two-action
    for s in range(n_states):
        # always better to act
        if arm_p_ucb[s, 0] > arm_p_ucb[s, 1]:  # move passive UCB down
            arm_p_ucb[s, 0] = arm_p_ucb[s, 1]
        if arm_p_lcb[s, 1] < arm_p_lcb[s, 1]:  # move active LCB up
            arm_p_lcb[s, 1] = arm_p_lcb[s, 1]

    assert n_states == 2  # these checks only valid for two-state
    for a in range(n_actions):
        # always better to start in good state
        if arm_p_ucb[0, a] > arm_p_ucb[1, a]:  # move bad-state UCB down
            arm_p_ucb[0, a] = arm_p_ucb[1, a]
        if arm_p_lcb[1, a] < arm_p_lcb[0, a]:  # move good-state LCB up
            arm_p_lcb[1, a] = arm_p_lcb[0, a]

    # these above corrections may lead to LCB being higher than UCBs... so make the UCB the optimistic option
    if arm_p_ucb[0, 0] < arm_p_lcb[0, 0]:
        print(f'ISSUE 00!! lcb {arm_p_lcb[0, 0]:.4f} ucb {arm_p_ucb[0, 0]:.4f}')
        arm_p_ucb[0, 0] = arm_p_lcb[0, 0] # p_ucb[i, 0, 0]
    if arm_p_ucb[0, 1] < arm_p_lcb[0, 1]:
        print(f'ISSUE 01!! lcb {arm_p_lcb[0, 1]:.4f} ucb {arm_p_ucb[0, 1]:.4f}')
        arm_p_ucb[0, 1] = arm_p_lcb[0, 1] # p_ucb[i, 0, 1]
    if arm_p_ucb[1, 0] < arm_p_lcb[1, 0]:
        print(f'ISSUE 10!! lcb {arm_p_lcb[1, 0]:.4f} ucb {arm_p_ucb[1, 0]:.4f}')
        arm_p_ucb[1, 0] = arm_p_lcb[1, 0] # p_ucb[i, 1, 0]
    if arm_p_ucb[1, 1] < arm_p_lcb[1, 1]:
        print(f'ISSUE 11!! lcb {arm_p_lcb[1, 1]:.4f} ucb {arm_p_ucb[1, 1]:.4f}')
        arm_p_ucb[1, 1] = arm_p_lcb[1, 1] # p_ucb[i, 1, 1]

    return arm_p_lcb, arm_p_ucb


def get_ucb_conf(cum_prob, n_pulls, t, alpha, episode_count, delta=1e-3,norm_confidence=False):
    """ calculate transition probability estimates """
    n_arms, n_states, n_actions = n_pulls.shape

    with np.errstate(divide='ignore'):
        n_pulls_at_least_1 = np.copy(n_pulls)
        n_pulls_at_least_1[n_pulls == 0] = 1
        est_p               = (cum_prob+1) / (n_pulls_at_least_1+n_states)
        # est_p               = (cum_prob) / (n_pulls_at_least_1)
        # est_p[n_pulls == 0] = 1 / n_states  # where division by 0

        if norm_confidence:
            z_score = norm.ppf(1-delta/2)
            conf_p = z_score*np.sqrt(est_p*(1-est_p)/n_pulls_at_least_1) 
        else: 
            conf_p = np.sqrt( 2 * n_states * np.log( 2 * n_states * n_actions * n_arms * ((episode_count+1)**4 / delta) ) / n_pulls_at_least_1 )
        conf_p[n_pulls == 0] = 1
        conf_p[conf_p > 1]   = 1  # keep within valid range 

    # if VERBOSE: print('conf', np.round(conf_p.flatten(), 2))
    # if VERBOSE: print('est p', np.round(est_p.flatten(), 2))

    return est_p, conf_p

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
