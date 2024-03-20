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

def process_zero_parameter_data(raw_data):
    """Turn a set of raw data, for all seeds + parameter combo
        Into a dictionary usable for plotting
        
    Arguments:
        raw_data: List of dictionaries for each parameter combo + seed
    
    Returns: Dictionary with averaged data per seed"""

    data_ret = {}

    keys = list(raw_data[0].keys())
    keys = [i for i in keys if i!='parameters']

    for k in keys:
        data_ret[k] = []

    for data in raw_data:
        for k in keys:
            data_ret[k].append(data[k])
    
    for k in keys:
        data_ret[k] = np.mean(data_ret[k],axis=0)
    return data


def process_one_parameter_data(raw_data,parameter_one):
    """Turn a set of raw data, for all seeds + parameter combo
        Into a dictionary usable for plotting
        
    Arguments:
        raw_data: List of dictionaries for each parameter combo + seed
    
    Returns: Dictionary with averaged data per seed"""

    data_by_arm_volunteer = {}

    for i in raw_data:
        n_arms = i['parameters'][parameter_one]

        if n_arms not in data_by_arm_volunteer:
            data_by_arm_volunteer[n_arms] = {}
        
    for n_arms in data_by_arm_volunteer:
        example_point = [i for i in raw_data 
            if i['parameters'][parameter_one] == n_arms]

        keys = list(example_point[0].keys())
        keys = [i for i in keys if i!='parameters']

        for k in keys:
            data_by_arm_volunteer[n_arms][k] = []

        for data in raw_data:
            if data['parameters'][parameter_one] == n_arms:
                for k in keys:
                    data_by_arm_volunteer[n_arms][k].append(data[k])
        
        for k in keys:
            data_by_arm_volunteer[n_arms][k] = np.mean(data_by_arm_volunteer[n_arms][k],axis=0)
    return data_by_arm_volunteer



def process_two_parameter_data(raw_data,parameter_one,parameter_two):
    """Turn a set of raw data, for all seeds + parameter combo
        Into a dictionary usable for plotting
        
    Arguments:
        raw_data: List of dictionaries for each parameter combo + seed
    
    Returns: Dictionary with averaged data per seed"""

    data_by_arm_volunteer = {}

    for i in raw_data:
        n_arms = i['parameters'][parameter_one]
        n_volunteers = i['parameters'][parameter_two]

        if n_arms not in data_by_arm_volunteer:
            data_by_arm_volunteer[n_arms] = {} 
        
        data_by_arm_volunteer[n_arms][n_volunteers] = {}

    for n_arms in data_by_arm_volunteer:
        for n_volunteers in data_by_arm_volunteer[n_arms]:
            example_point = [i for i in raw_data 
                if i['parameters'][parameter_one] == n_arms and i['parameters'][parameter_two] == n_volunteers]

            keys = list(example_point[0].keys())
            keys = [i for i in keys if i!='parameters']

            for k in keys:
                data_by_arm_volunteer[n_arms][n_volunteers][k] = []

            for data in raw_data:
                if data['parameters'][parameter_one] == n_arms and \
                data['parameters'][parameter_two] == n_volunteers:
                    for k in keys:
                        data_by_arm_volunteer[n_arms][n_volunteers][k].append(data[k])
            
            for k in keys:
                data_by_arm_volunteer[n_arms][n_volunteers][k] = np.mean(data_by_arm_volunteer[n_arms][n_volunteers][k],axis=0)
    return data_by_arm_volunteer

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
    
    for i in ret_dict:
        ret_dict[i] = (np.mean(ret_dict[i]),np.std(ret_dict[i]))
    
    return ret_dict 

def plot_tradeoff_curve(data,names,nice_names,title):
    """Plot the matching vs. activity tradeoff curve
    
    Arguments:
        data: Dictionary of list of data for each policy parameter
        names: List of strings, such as 'whittle', so that
            whittle_match and whittle_activity are in data
        nice_names: List the same size as policies with 
            nicer names for the plot legend
    
    Returns: Nothing
    
    Side Effects: Plots data"""

    fig, ax = plt.subplots(figsize=(5,7))
    ax.set_title(title)

    color_palette = plt.cm.viridis(np.linspace(0, 1, len(names)))

    i = 0
    for name,nice_name in zip(names,nice_names):
        print(name,nice_name)
        data_points = list(sorted(zip(data['{}_active'.format(name)],data['{}_match'.format(name)]),key=lambda k: k[0]))
        x,y = zip(*data_points)
        plt.plot(x,y,label=nice_name,color=color_palette[i])
        plt.scatter(x,y,color=color_palette[i])

        i+=1 

    plt.xlabel("Active Rate")
    plt.ylabel("Match Rate")
    plt.legend() 