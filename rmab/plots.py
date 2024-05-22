import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from copy import deepcopy

def plot_transition_dynamics(transitions):
    """Plot the transition dynamics, given a 2x2x2 matrix transitions
    
    Arguments: 
        transitions: 2x2x2 numpy matrix
    
    Returns: The colorbar, so that the ticks can be changed 
    
    Side Effects: Plots a 2x2x2 numpy matrix in heatmap"""

    plt.figure(figsize=(4,3))
    ax = sns.heatmap(transitions[:,:,1],xticklabels=['No Notif.','Notif.'],yticklabels=['Inactive','Active'])
    plt.xlabel("Action",fontsize=16)
    plt.ylabel("Start State",fontsize=16)
    cbar = ax.collections[0].colorbar
    cbar.set_label("Active probability", rotation=270, labelpad=20)

    return cbar 

def aggregate_data(results):
    """Get the average and standard deviation for each key across 
        multiple trials
        
    Arguments: 
        results: List of dictionaries, one for each seed
    
    Returns: Dictionary, with each key mapping to a 
        tuple with the mean and standard deviation"""

    ret_dict = {}
    for l in results:
        for k in l:
            if type(l[k]) == int or type(l[k]) == float:
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k].append(l[k])
            elif type(l[k]) == list and (type(l[k][0]) == int or type(l[k][0]) == float):
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k].append(l[k][0])
            elif type(l[k]) == type(np.array([1,2])):
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k] += list(l[k])

    for i in ret_dict:
        ret_dict[i] = (np.mean(ret_dict[i]),np.std(ret_dict[i]))
    
    return ret_dict 

def aggregate_normalize_data(results,baseline=None):
    """Get the average and standard deviation for each key across 
        multiple trials; with each reward/etc. being averaged
        
    Arguments: 
        results: List of dictionaries, one for each seed
    
    Returns: Dictionary, with each key mapping to a 
        tuple with the mean and standard deviation"""

    results_copy = deepcopy(results)

    for data_point in results_copy:
        avg_by_type = {}
        linear_whittle_results = {}
        for key in data_point:
            is_list = False
            if type(data_point[key]) == list and (type(data_point[key][0]) == int or type(data_point[key][0]) == float):
                value = data_point[key][0]
            elif type(data_point[key]) == int or type(data_point[key]) == float:
                value = data_point[key]
            elif type(data_point[key]) == list and type(data_point[key][0]) == list:
                is_list = True 
                value = data_point[key][0]
                data_point[key] = np.array(data_point[key][0])
            else:
                continue 
            data_type = key.split("_")[-1]
            if data_type not in avg_by_type and key == "{}_{}".format(baseline,data_type):
                if is_list:
                    avg_by_type[data_type] = np.array(data_point[key])
                else:
                    avg_by_type[data_type] = data_point[key][0]
        if baseline != None:
            for key in data_point:
                data_type = key.split("_")[-1]
                if data_type in avg_by_type:
                    if type(avg_by_type[data_type]) == type(np.array([1,2])):
                        data_point[key] = data_point[key]/avg_by_type[data_type]
                        data_point[key] -= 1
                    else:
                        data_point[key][0] /= float(avg_by_type[data_type])

    return aggregate_data(results_copy)