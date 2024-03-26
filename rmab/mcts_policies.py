import numpy as np
import random 

from rmab.uc_whittle import Memoizer
from rmab.omniscient_policies import whittle_index

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import gc 
import time
from sklearn.cluster import KMeans

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,use_sigmoid=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.use_sigmoid = use_sigmoid 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        if self.use_sigmoid:
            x = self.sigmoid(x)

        return x

def epsilon_func(n):
    """Probability of selecting a random volunteer, given # past things seen
    Arguments:
        n: Number of past runs seen
    Returns: Float, probability of selecting a random volunteer
    """

    return 1/(n/5+1)**.5

def epsilon_func_index(n):
    return 1/(n+1)**0.1

def get_random_index(env,all_match_probs,best_group_arms,policy_by_group,group_indices,num_epochs,lamb,memoizer,state,budget,state_WI,randomness_boost=0.2):
    """Leverage the policy network and heuristics to select a random arm

    Arguments:
        env: RMAB Simulator Environment
        all_match_probs: List of match probabilities for each of the N agents
        best_group_arms: List of lists; best_group_arms[i][j] is the jth best arm
            in group i (according to the match probability)
            This assumes that all arms within a group have the same transition
        policy_by_group: List of lists; policy_by_group[i][j] is the probability
            predicted by the policy of pulling the corresponding arm in 
            best_group_arms
        group_indices: How many we've selected from each group  
            List of length num_groups
        num_epochs: How many epochs have we train value networks, etc. for
    
    Returns: Integer, which group to select next
    """

    score_by_group = [] 
    score_by_group_policy = []

    num_groups = len(group_indices)

    for g in range(num_groups):
        if group_indices[g] >= len(best_group_arms[g]):
            score_by_group.append(0)
            score_by_group_policy.append(0)
        else:
            score_by_group.append(state_WI[best_group_arms[g][group_indices[g]]]+randomness_boost)
            score_by_group_policy.append(max(policy_by_group[g][group_indices[g]],0.001))

    fraction_policy = 1 - epsilon_func_index(num_epochs)
    sum_group = sum(score_by_group)
    sum_policy = sum(score_by_group_policy)

    weighted_probabilities = []
    for i in range(len(score_by_group_policy)):
        if score_by_group[i] == 0 and score_by_group_policy[i] == 0:
            prob = 0
        else:
            prob = fraction_policy*score_by_group_policy[i]/sum_policy
            prob += (1-fraction_policy)/2*score_by_group[i]/sum_group + (1-fraction_policy)/2*1/(len(score_by_group_policy))
        weighted_probabilities.append(prob)
    weighted_probabilities = np.array(weighted_probabilities)/np.sum(weighted_probabilities)

    # With some fixed probability, pick randomly 
    random_prob = 0.3
    if random.random() < random_prob:
        weighted_probabilities = [1 if group_indices[i] < len(best_group_arms[i]) else 0 for i in range(len(score_by_group))]
        weighted_probabilities = np.array(weighted_probabilities)/np.sum(weighted_probabilities)
    
    selected_index = random.choices(range(len(score_by_group)), weights=weighted_probabilities, k=1)[0]
    return selected_index, memoizer 

def group_index_to_action(best_group_arms,group_indices,state):
    """Given a list of # of each group to pull, convert this to a 0-1 list
    
    Arguments:
        best_group_arms: The best arms for each group
        group_indices: How much to pull each group
        
    Returns: 0-1 List of whether each arm is pulled"""

    ret_list = np.zeros(len(state))
    for g in range(len(best_group_arms)):
        for i in range(group_indices[g]):
            ret_list[best_group_arms[g][i]] = 1
    
    return ret_list 

def get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,policy_network,lamb,num_future_samples=25,
                    weighted=False,contextual=False):
    """Leverage the value network and reward to get the total value for an action

    Arguments:
        env: RMAB Simulator Environment
        all_match_probs: List of match probabilities for each of the N agents
        best_group_arms: List of lists; best_group_arms[i][j] is the jth best arm
            in group i (according to the match probability)
            This assumes that all arms within a group have the same transition
        state: 0-1 list; which state is each arm in
        group_indices: How many we've selected from each group  
            List of length num_groups
        value_network: PyTorch model that maps states to their value (float)
    
    Returns: Float, value of the current state (including current + discounted future value)
    """
    non_match_prob = 1
    num_groups = len(best_group_arms)

    for g in range(num_groups):
        for i in range(group_indices[g]):
            if contextual: 
                match_probs = env.current_episode_match_probs[env.timestep + env.episode_count*env.episode_len][env.agent_idx]
                non_match_prob *=  1-match_probs[best_group_arms[g][i]]*state[best_group_arms[g][i]]
                
            else:
                non_match_prob *= 1-all_match_probs[best_group_arms[g][i]]*state[best_group_arms[g][i]]

    match_prob = 1-non_match_prob
    action = np.zeros(len(state), dtype=np.int8)
    for g in range(num_groups):
        for j in range(group_indices[g]):
            action[best_group_arms[g][j]] = 1

    probs = []
    for i in range(len(state)):
        probs.append(env.transitions[i//env.volunteers_per_arm, state[i], action[i], 1])

    samples = np.zeros((num_future_samples,len(state)))
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            samples[i,j] = random.random() 
    samples = samples < probs 
    samples = samples.astype(float)

    future_actions = get_action_state(policy_network,samples,env,contextual=contextual)

    future_actions = torch.Tensor(future_actions)
    future_states = torch.Tensor(samples)
    combined = torch.cat((future_states, future_actions), dim=1)

    future_values = value_network(combined).detach().numpy()  

    if weighted:
        weights = np.array([np.prod([p if xi else 1 - p for xi, p in zip(x, probs)]) for x in samples]).reshape((len(samples),1))
        weights /= np.sum(weights)
        future_value = np.sum(future_values*weights)
    else:
        future_value = np.mean(future_values)
    total_value = match_prob*(1-lamb) + np.mean(state)*lamb+future_value * env.discount 

    return total_value 

def update_value_function(past_states,past_actions,value_losses,value_network,criterion_value,optimizer_value,
                                    policy_network,env,lamb,weighted=False,contextual=False):
    """Run backprop on the Value and Policy Networks
    
    Arguments:
        past_states: Feature dataset; list of previous state seen
        past_values: For each past state seen, what was the computed value? 
        value_network: PyTorch model mapping states to float values
        criterion_value: MSE Loss, used for loss computation
        optimizer_value: SGD Optimizer for the value network
        policy_networks: List of PyTorch models mapping states to action probabilities
        criterion_policy: BCE Loss, used for loss computation
        optimizers_policy: List of SGD Optimizer for the policy network
        value_losses: List of losses from the value network; used for debugging
        policy_losses: List of losses from the policy network; used for debugging
    
    Returns: Nothing

    Side Effects: Runs backpropogation on value_network
        using a random previously seen point
    
    """

    random_point = max(len(past_states)-random.randint(1,200),0) # len(past_states)-1
    state = past_states[random_point]
    action = past_actions[random_point]
    non_match_prob = 1

    if contextual:
        match_probabilities = env.current_episode_match_probs[:,env.agent_idx]
        all_match_probs = np.mean(match_probabilities,axis=0)
    else:
        all_match_probs = np.array(env.match_probability_list)[env.agent_idx]
        for i in range(len(action)):
            non_match_prob *= np.prod(1-all_match_probs[i]*action[i]*state[i])

    samples = np.zeros((25,len(state)))
    for i in range(samples.shape[0]):
        for j in range(samples.shape[1]):
            samples[i,j] = random.random() 
    probs = []
    for i in range(len(state)):
        probs.append(env.transitions[i//env.volunteers_per_arm, state[i], action[i], 1])
    samples = samples < np.array(probs) 
    samples = samples.astype(float)
    if weighted:
        weights = np.array([np.prod([p if xi else 1 - p for xi, p in zip(x, probs)]) for x in samples]).reshape((len(samples),1))
        weights /= np.sum(weights)

    future_actions = get_action_state(policy_network,samples,env,contextual=contextual)
    future_actions = torch.Tensor(future_actions)

    future_states = torch.Tensor(samples)
    combined =  torch.cat((future_states, future_actions), dim=1)
    future_values = value_network(combined).detach().numpy()   
    
    # if len(past_states) > 500:
    #     print("Future Values are {}".format(np.mean(future_values)))

    future_values *= env.discount

    future_values += (1-non_match_prob)*(1-lamb) + np.mean(state)*lamb

    # if len(past_states) > 500:
    #     print("Present values are {}".format((1-non_match_prob)*(1-lamb) + np.mean(state)*lamb))
    #     print("Total values are {}".format(np.mean(future_values)))

    target = torch.Tensor(future_values)
    input_data = torch.Tensor([list(past_states[random_point])+list(past_actions[random_point]) for i in range(25)])
    output = value_network(input_data)

    loss_value = criterion_value(output,target)
    optimizer_value.zero_grad()
    loss_value.backward()
    optimizer_value.step()
    value_losses.append(torch.mean(loss_value).item())

def update_policy_function(past_states,past_actions,policy_losses,policy_network,
            criterion_policy,optimizer_policy,env,contextual=False):
    """Run backprop on the Value and Policy Networks
    
    Arguments:
        past_states: Feature dataset; list of previous state seen
        past_values: For each past state seen, what was the computed value? 
        value_network: PyTorch model mapping states to float values
        criterion_value: MSE Loss, used for loss computation
        optimizer_value: SGD Optimizer for the value network
        policy_networks: List of PyTorch models mapping states to action probabilities
        criterion_policy: BCE Loss, used for loss computation
        optimizers_policy: List of SGD Optimizer for the policy network
        value_losses: List of losses from the value network; used for debugging
        policy_losses: List of losses from the policy network; used for debugging
    
    Returns: Nothing

    Side Effects: Runs backpropogation on value_network
        using a random previously seen point
    
    """
    
    random_point = max(len(past_states)-random.randint(1,200),0) 
    target = torch.Tensor(past_actions[random_point])    
    x_points = torch.Tensor(get_policy_network_input(env,past_states[random_point],contextual=contextual))

    output = policy_network(x_points)
    loss_policy = criterion_policy(output,target.reshape(len(target),1))

    optimizer_policy.zero_grad()  
    loss_policy.backward()  
    optimizer_policy.step() 
    total_policy_loss = loss_policy.item()
    policy_losses.append(total_policy_loss)

def get_best_combo(last_prefixes):
    """Using the data we've seen from the MCTS run, determine which combo of arms to pull
    
    Arguments:
        last_prefixes: Dictionary mapping arms pulled to value
        budget: Integer, total number of arms to pull
    
    Returns: List, best combination of arms to pull"""

    combo_list = [] 
    for prefix in last_prefixes:
        value = np.mean(last_prefixes[prefix])
        combo = eval(prefix) 
        combo = sorted(combo)
        combo_list.append((combo,value))

    sorted_combos = sorted(combo_list,key=lambda k: k[1])[-1]
    return sorted_combos[0]

def get_policy_network_input_many_state(env,state_list,contextual=False):
    """From an environment + state, construct the input for the policy function
    
    Arguments:
        env: RMAB Simulator Environment
        state: 0-1 numpy array
    
    Returns: List of x_points, which can be turned into a Tensor 
        for the policy network"""

    if not contextual:
        all_match_probs = np.array(env.match_probability_list)[env.agent_idx]
    
    policy_network_size = 6+len(state_list[0])
    if contextual:
        policy_network_size = 6+len(state_list[0])    
        
    x_points = np.zeros((state_list.size,policy_network_size))

    for i in range(len(state_list[0])):
        transition_points = env.transitions[i//env.volunteers_per_arm][:,:,1].flatten() 
        x_points[i::len(state_list[0]),:len(transition_points)] = transition_points
    
        if contextual:
            x_points[i::len(state_list[0]),len(transition_points)] = env.current_episode_match_probs[
                env.timestep + env.episode_count*env.episode_len,env.agent_idx[i]]
        else:
            x_points[i::len(state_list[0]),len(transition_points)] = all_match_probs[i]
    
    shift = 1
    if contextual:
        shift = 1 

    x_points[:,len(transition_points)+shift] = state_list.flatten() 
    x_points[:,len(transition_points)+shift+1:] = np.repeat(state_list,len(state_list[0]),axis=0)
    
    return x_points 


def get_policy_network_input(env,state,contextual=False):
    """From an environment + state, construct the input for the policy function
    
    Arguments:
        env: RMAB Simulator Environment
        state: 0-1 numpy array
    
    Returns: List of x_points, which can be turned into a Tensor 
        for the policy network"""

    if not contextual:
        all_match_probs = np.array(env.match_probability_list)[env.agent_idx]

    x_points = []
    for i in range(len(state)):
        transition_points = env.transitions[i//env.volunteers_per_arm][:,:,1].flatten() 
        if contextual:
            x_points.append(list(transition_points)+[env.current_episode_match_probs[
                env.timestep + env.episode_count*env.episode_len,env.agent_idx[i]]]+[state[i]]+list(state))
        else:
            x_points.append(list(transition_points)+list([all_match_probs[i]])+[state[i]]+list(state))
    
    return x_points 

def get_action_state(policy_network,state_list,env,contextual=False):
    """Given a state, find the best action using the policy network
    
    Arguments: 
        policy_network: MLP that predicts the best action for each agent
        state_list: Numpy matrix of 0-1 states
        Environment: RMABSimulator Environment
    
    Returns: Numpy array, 0-1 action for each agent"""

    x_points = get_policy_network_input_many_state(env,np.array(state_list),contextual=contextual)

    policy_network_predictions = policy_network(torch.Tensor(x_points)).detach() 
    policy_network_predictions = policy_network_predictions.numpy()

    action = np.zeros(np.array(state_list).shape, dtype=np.int8)

    for i in range(len(state_list)):
        action[i][np.argsort(policy_network_predictions[i*len(state_list[0]):(i+1)*len(state_list[0])].flatten())[::-1][:env.budget]] = 1
    return action 

def action_to_group_combo(action,best_group_arms):
    """Convert an action for each user into a list of 
        groups pulled
        
    Arguments:
        action: 0-1 Numpy array
        best_grou_arms: List of lists; best arms for each group
        
    Returns: List of groups pulled (including duplicates)"""

    group_indices = [0 for i in range(len(best_group_arms))]
    full_combo = []
    for g in range(len(best_group_arms)):
        for _,i in enumerate(best_group_arms[g]):
            if action[i] == 1:
                group_indices[g] += 1
                full_combo.append(g)
    full_combo = sorted(full_combo)

    return full_combo, group_indices

def full_mcts_policy_contextual(env,state,budget,lamb,memory,per_epoch_results):
    return full_mcts_policy(env,state,budget,lamb,memory,per_epoch_results,contextual=True,group_setup="none")

def full_mcts_policy_contextual_rand_group(env,state,budget,lamb,memory,per_epoch_results):
    return full_mcts_policy(env,state,budget,lamb,memory,per_epoch_results,contextual=True,group_setup="random")

def full_mcts_policy_contextual_transition_group(env,state,budget,lamb,memory,per_epoch_results):
    return full_mcts_policy(env,state,budget,lamb,memory,per_epoch_results,contextual=True,group_setup="transition")

def full_mcts_policy_contextual_whittle_group(env,state,budget,lamb,memory,per_epoch_results):
    return full_mcts_policy(env,state,budget,lamb,memory,per_epoch_results,contextual=True,group_setup="whittle")

def full_mcts_policy(env,state,budget,lamb,memory,per_epoch_results,contextual=False,group_setup="transition"):
    """Compute an MCTS policy by first splitting up into groups
        Then running MCTS, bootstrapped by a policy pi and a value V
        Upate pi, V, then return the best action

    Arguments:
        env: Simulator Environment
        state: Num Agents x 2 numpy array (0-1)
        budget: Integer, how many arms we can pull
        Lamb: Balance between engagement, global reward
        Memory: Contains the V, Pi network
        per_epoch_results: Optional argument, nothing for this 
        timeLimit: Optional, max time we can run for
    """

    num_iterations = env.mcts_train_iterations
    value_lr = env.value_lr 
    policy_lr = env.policy_lr 
    past_prefixes = {}
    last_prefixes = {}
    best_score = 0
    train_epochs = env.train_epochs
    network_update_frequency = 25
    num_network_updates = 25
    randomness_boost = 0.2 

    if not contextual:
        all_match_probs = np.array(env.match_probability_list)[env.agent_idx]
    else:
        all_match_probs = [0 for i in range(len(env.agent_idx))]

    if len(state) > 1000:
        gc.collect() 

    # Initialize the memory 
    if memory == None:
        past_states = []
        past_values = []
        past_actions = []
        value_losses = []
        policy_losses = []
        value_network = MLP(len(state)*2, 512, 1)
        criterion_value = nn.MSELoss()
        optimizer_value = optim.SGD(value_network.parameters(), lr=value_lr)
        criterion_policy = nn.BCELoss()
        memoizer = Memoizer('optimal')
        policy_network_size = 6+len(state)
        if contextual:
            policy_network_size = 6+len(state) 
        policy_network = MLP(policy_network_size,128,1,use_sigmoid=True)
        optimizer_policy = optim.Adam(policy_network.parameters(),lr=policy_lr)

        if contextual:
            match_probabilities = env.current_episode_match_probs[:,env.agent_idx]
            match_probs = np.mean(match_probabilities,axis=0)
            best_group_arms = [] 
        else:
            whittle_index(env,[0 for i in range(len(all_match_probs))],budget,lamb,memoizer)
            whittle_index(env,[1 for i in range(len(all_match_probs))],budget,lamb,memoizer)
    else:
        if contextual:
            other_info = memory[-1] 
            match_probs = other_info['match_probs']
            best_group_arms = other_info['best_group_arms']
            memory = memory[:-1]
        past_states, past_values, past_actions, \
        value_losses, value_network, criterion_value, optimizer_value, \
        policy_losses, policy_network, criterion_policy, optimizer_policy, \
            memoizer = memory
    num_epochs = len(past_states)

    # Step 1: Group Setup
    if group_setup == "none":
        num_groups = len(state) 
        best_group_arms = []
        for g in range(num_groups):
            best_group_arms.append([g])
    elif group_setup == "random":
        if best_group_arms == []:
            num_groups = round(len(state)**.5)
            elements_per_group = len(state)//num_groups 
            all_elements = list(range(len(state)))
            random.shuffle(all_elements)
            best_group_arms = []

            for i in range(num_groups):
                if i == num_groups-1:
                    best_group_arms.append(all_elements[elements_per_group*i:])
                else:
                    best_group_arms.append(all_elements[elements_per_group*i:elements_per_group*(i+1)])
        num_groups = len(best_group_arms)
    elif group_setup == "transition":
        num_groups = len(state)//env.volunteers_per_arm * 2 
        best_group_arms = []
        for g in range(num_groups//2):
            if contextual: 
                matching_probs = match_probs[g*env.volunteers_per_arm:(g+1)*env.volunteers_per_arm]
            else:
                matching_probs = all_match_probs[g*env.volunteers_per_arm:(g+1)*env.volunteers_per_arm]
            sorted_matching_probs = np.argsort(matching_probs)[::-1] + g*env.volunteers_per_arm
            sorted_matching_probs_0 = [i for i in sorted_matching_probs if state[i] == 0]
            sorted_matching_probs_1 = [i for i in sorted_matching_probs if state[i] == 1]
            best_group_arms.append(sorted_matching_probs_0)
            best_group_arms.append(sorted_matching_probs_1)
    elif group_setup == "whittle":
        if best_group_arms == []:
            num_groups = round(len(state)**.5)
            if contextual:
                state_WI = whittle_index(env,[1 for i in range(len(state))],budget,lamb,memoizer,contextual=contextual,match_probs=match_probs)
            else:
                state_WI = whittle_index(env,state,budget,lamb,memoizer)
            kmeans = KMeans(n_clusters=num_groups)
            state_WI = np.array(state_WI).reshape(-1,1)
            kmeans.fit(state_WI)
            cluster_labels = kmeans.labels_
            best_group_arms = [[] for i in range(num_groups)]

            for i in range(len(cluster_labels)):
                best_group_arms[cluster_labels[i]].append(i)

        num_groups = len(best_group_arms)


    # Step 2: MCTS iterations
    # Get the probability for pulling each arm, in the same order
    # As the 'best_group_arms' variable
    action = get_action_state(policy_network,[state],env,contextual=contextual)
    x_points = get_policy_network_input(env,state,contextual=contextual)
    policy_network_predictions = np.array(policy_network(torch.Tensor(x_points)).detach().numpy().T)[0]
    policy_by_group = []
    for g in range(num_groups):
        policy_by_group.append(policy_network_predictions[best_group_arms[g]])

    action = np.zeros(len(state), dtype=np.int8)
    action[np.argsort(policy_network_predictions)[::-1][:budget]] = 1
    full_combo, group_indices = action_to_group_combo(action,best_group_arms)
    total_value = get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,policy_network,lamb,num_future_samples=25,contextual=contextual)
    best_score = max(best_score,total_value)
    full_combo_string = repr(full_combo)
    last_prefixes[full_combo_string] = [total_value]

    if num_epochs >= train_epochs:
        num_iterations = env.mcts_test_iterations

    # Used to efficiently compute heuristics 
    if contextual: 
        state_WI = whittle_index(env,state,budget,lamb,memoizer,contextual=contextual,match_probs=match_probs)
    else:
        state_WI = whittle_index(env,state,budget,lamb,memoizer)

    action = np.zeros(len(state), dtype=np.int8)
    action[np.argsort(state_WI)[::-1][:budget]] = 1
    full_combo, group_indices = action_to_group_combo(action,best_group_arms)
    total_value = get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,policy_network,lamb,num_future_samples=25,contextual=contextual)
    best_score = max(best_score,total_value)
    full_combo_string = repr(full_combo)
    last_prefixes[full_combo_string] = [total_value]

    # Main MCTS loop
    for _ in range(num_iterations):
        current_combo = [] 
        group_indices = [0 for i in range(num_groups)]
        update_combos = []
        skip_iteration = False

        for k in range(budget): 
            # Find past runs with same arm combinations 
            scores_current_combo = {}
            if repr(current_combo) in past_prefixes:
                scores_current_combo = past_prefixes[repr(current_combo)]
            n = len(scores_current_combo)
    
            # Find upper bounds to determine if it's worthwhile to keep exploring 
            UCB_by_arm = {}
            should_random = False 
            should_break = False 
            if len(scores_current_combo) > 0:
                current_value = get_total_value(env,all_match_probs,best_group_arms,state,
                                                group_indices,value_network,policy_network,lamb,num_future_samples=25,contextual=contextual)
                for arm in scores_current_combo:
                    new_group_index = deepcopy(group_indices)
                    new_group_index[arm] += 1
                    value_with_pull = get_total_value(env,all_match_probs,best_group_arms,state,new_group_index,value_network,policy_network,lamb,num_future_samples=10,contextual=contextual)
                    value_with_pull *= (1+0.1) # Estimation error 
                    upper_bound = current_value + (value_with_pull-current_value)*(budget-k)
                    UCB_by_arm[arm] = upper_bound 
                    
                
                if max(UCB_by_arm.values()) < best_score:
                    if len(UCB_by_arm) == num_groups:
                        should_break = True 
                    else:
                        should_random = True 

            if should_break:
                skip_iteration = True
                break 

            # Determine which arm to explore
            if not should_random:
                should_random = random.random() <= epsilon_func(n)

            if should_random:
                selected_index, memoizer = get_random_index(env,all_match_probs,best_group_arms,policy_by_group,group_indices,num_epochs,lamb,memoizer,state,budget,state_WI,randomness_boost)
            else:
                selected_index = max(UCB_by_arm, key=UCB_by_arm.get)
            update_combos.append((current_combo[:],selected_index))
            current_combo.append(selected_index)
            group_indices[selected_index] += 1 

        if skip_iteration:
            continue 
    
        should_break = False 

        # Get the total value, and update this along subsets of arms pulled 
        total_value = get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,policy_network,lamb,num_future_samples=25,contextual=contextual)
        best_score = max(best_score,total_value)
        for (prefix,next_index) in update_combos:
            prefix_string = repr(prefix)
            full_combo = sorted(prefix+[next_index])
            full_combo_string = repr(full_combo)
            if prefix_string not in past_prefixes:
                past_prefixes[prefix_string] = {} 
            if next_index not in past_prefixes[prefix_string]:
                past_prefixes[prefix_string][next_index] = 0
            past_prefixes[prefix_string][next_index] = max(past_prefixes[prefix_string][next_index],total_value)
            if len(prefix) == budget-1: 
                if repr(full_combo) not in last_prefixes:
                    last_prefixes[full_combo_string] = [] 
                last_prefixes[full_combo_string].append(total_value)
                if len(last_prefixes) > 10:
                    should_break = True
        if should_break:
            break 
        
    # Step 3: Find the best action/value from all combos, and backprop 
    # Find the best action/value combo 
    best_combo = get_best_combo(last_prefixes)
    combo_to_volunteers = []
    group_indices = [0 for i in range(num_groups)]
    for i in best_combo:
        combo_to_volunteers.append(best_group_arms[i][group_indices[i]])
        group_indices[i] += 1
    best_value = get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,policy_network,lamb,num_future_samples=25,contextual=contextual)
    action = np.zeros(len(state), dtype=np.int8)
    action[combo_to_volunteers] = 1

    # Update datasets + backprop 
    past_states.append(list(state))
    past_values.append(best_value)
    past_actions.append(action)

    if num_epochs < train_epochs and (num_epochs+1)%network_update_frequency == 0:
        for i in range(num_network_updates):
            update_value_function(past_states,past_actions,value_losses,value_network,criterion_value,optimizer_value,
                                    policy_network,env,lamb,contextual=contextual)

            update_policy_function(past_states,past_actions,policy_losses,policy_network,
            criterion_policy,optimizer_policy,env,contextual=contextual)
    if contextual:
        memory = past_states, past_values, past_actions, value_losses, value_network, criterion_value, optimizer_value, policy_losses, policy_network, criterion_policy, optimizer_policy, memoizer, {'match_probs': match_probs, 'best_group_arms': best_group_arms} 
    else:
        memory = past_states, past_values, past_actions, value_losses, value_network, criterion_value, optimizer_value, policy_losses, policy_network, criterion_policy, optimizer_policy, memoizer 

    return action, memory 