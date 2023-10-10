import matplotlib.pyplot as plt 
import seaborn as sns

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