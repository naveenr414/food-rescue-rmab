import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

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

def plot_line_plot_parameter(data,parameter,target_var,method_to_nice):
    """Plot the values for a parameter, given a dataset where the parameter varies

    Arguments:
        data: List of dictionaries, each with 'parameter' key
        parameter: String, some parameter in the dataset
        target_var: String, which value we're trying to understand 
        method_to_nice: Dictionary, convert method names to nicer names

    Returns: Nothing

    Side Effects: Plots the data, along with standard deviations
    """

    data_by_method = {}

    for i in data:
        param_value = i['parameters'][parameter]
        methods = i[target_var].keys() 

        if data_by_method == {}:
            for m in methods:
                data_by_method[m] = {}
        
        for m in methods:
            data_by_method[m][param_value] = []

    for i in data:
        param_value = i['parameters'][parameter]

        for m in i[target_var]:
            data_by_method[m][param_value].append(i[target_var][m])
    
    for m in data_by_method:
        keys = sorted(list(data_by_method[m].keys()))
        mean_vals = np.array([np.mean(data_by_method[m][k]) for k in keys])
        std_vals = np.array([np.std(data_by_method[m][k]) for k in keys])

        plt.plot(keys,mean_vals,label=method_to_nice[m])
        plt.fill_between(keys, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3)