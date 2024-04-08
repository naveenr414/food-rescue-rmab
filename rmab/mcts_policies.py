import numpy as np
import random 

from rmab.uc_whittle import Memoizer
from rmab.omniscient_policies import whittle_index, shapley_index_custom, shapley_whittle_custom_policy
from rmab.utils import custom_reward, binary_to_decimal, list_to_binary
from rmab.compute_whittle import get_q_vals, arm_compute_whittle, arm_compute_whittle_multi_prob

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import gc 
import time
from sklearn.cluster import KMeans
import torch.nn.functional as F
from collections import defaultdict 
import math 
import torch.optim.lr_scheduler as lr_scheduler

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,use_sigmoid=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.use_sigmoid = use_sigmoid 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

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

def get_random_index(env,all_match_probs,best_group_arms,policy_by_group,group_indices,num_epochs,lamb,memoizer,state,budget,state_WI,randomness_boost=0.2,use_whittle=True):
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
            if use_whittle:
                score_by_group.append(state_WI[best_group_arms[g][group_indices[g]]]+randomness_boost)
            else:
                score_by_group.append(0)
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
            if use_whittle:
                prob += (1-fraction_policy)/2*score_by_group[i]/sum_group
            prob += (1-fraction_policy)/2*1/(len(score_by_group_policy))
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
                    weighted=False,contextual=False,memoizer=None):
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

    future_actions = get_action_state(policy_network,samples,env,contextual=contextual,memoizer=memoizer)

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
                                    policy_network,env,lamb,weighted=False,contextual=False,memoizer=None):
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

    future_actions = get_action_state(policy_network,samples,env,contextual=contextual,memoizer=memoizer)
    future_actions = torch.Tensor(future_actions)

    future_states = torch.Tensor(samples)
    combined =  torch.cat((future_states, future_actions), dim=1)
    future_values = value_network(combined).detach().numpy()   
    
    future_values *= env.discount

    future_values += (1-non_match_prob)*(1-lamb) + np.mean(state)*lamb

    target = torch.Tensor(future_values)
    input_data = torch.Tensor([list(past_states[random_point])+list(past_actions[random_point]) for i in range(25)])
    output = value_network(input_data)

    loss_value = criterion_value(output,target)
    optimizer_value.zero_grad()
    loss_value.backward()
    optimizer_value.step()
    value_losses.append(torch.mean(loss_value).item())

def update_policy_function(past_states,past_actions,policy_losses,policy_network,
            criterion_policy,optimizer_policy,env,contextual=False,memoizer=None):
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
    x_points = torch.Tensor(get_policy_network_input_many_state(env,np.array([past_states[random_point]]),contextual=contextual,memoizer=memoizer))

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

def get_policy_network_input_many_state(env,state_list,contextual=False,memoizer=None):
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


def get_action_state(policy_network,state_list,env,contextual=False,memoizer=None):
    """Given a state, find the best action using the policy network
    
    Arguments: 
        policy_network: MLP that predicts the best action for each agent
        state_list: Numpy matrix of 0-1 states
        Environment: RMABSimulator Environment
    
    Returns: Numpy array, 0-1 action for each agent"""

    x_points = get_policy_network_input_many_state(env,np.array(state_list),contextual=contextual,memoizer=memoizer)

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

def get_groups(group_setup,best_group_arms,state,env,match_probs,all_match_probs,contextual,lamb,memoizer,budget):
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
    
    return best_group_arms

from collections import namedtuple, deque
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def get_reward(state,action,match_probs,lamb):
    prod_state = 1-state*action*np.array(match_probs)
    prob_all_inactive = np.prod(prod_state)
    return (1-prob_all_inactive)*(1-lamb) + np.sum(state)/len(state)*lamb

def get_reward_max(state,action,match_probs,lamb):
    prod_state = state*action*np.array(match_probs)
    score = np.max(prod_state)
    return score*(1-lamb) + np.sum(state)/len(state)*lamb

def get_reward_custom(state,action,match_probs,lamb):
    return custom_reward(state,action,match_probs)*(1-lamb) + np.sum(state)/len(state)*lamb


def dqn_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",stabilization=False):
    """Use a DQN policy to compute the action values
    
    Arguments: 
        env: Simulator Environment
        state: Num Agents x 2 numpy array (0-1)
        budget: Integer, how many arms we can pull
        Lamb: Balance between engagement, global reward
        Memory: Contains the V, Pi network
        per_epoch_results: Optional argument, nothing for this 
    Returns: Numpy array, action"""

    # Hyperparameters + Parameters 
    
    value_lr = 5e-4
    train_epochs = env.train_epochs
    N = len(state) 
    match_probs = np.array(env.match_probability_list)[env.agent_idx]
    epsilon = 0.01
    target_update_freq = 1
    batch_size = 16
    discount = env.discount
    MAX_REPLAY_SIZE = 512
    tau = 0.01
    valid_actions = []
    action_to_idx = {}
    for action_num in range(2**(N)):
        action = [int(j) for j in bin(action_num)[2:].zfill(N)]
        if sum(action) <= budget:
            valid_actions.append(action)
            action_to_idx[''.join([str(i) for i in action])] = len(valid_actions)-1

    # Unpack the memory 
    if memory == None:
        past_states = []
        past_actions = []
        past_rewards = []
        past_probs = []
        q_losses = []
        q_network = MLP(len(state), 128, len(valid_actions))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(q_network.parameters(), lr=value_lr)
        q_network.fc3.bias = torch.nn.Parameter(torch.Tensor([round(env.avg_reward) for i in range(len(q_network.fc3.bias))]))
        target_model = MLP(len(state), 128, len(valid_actions))
        target_model.load_state_dict(q_network.state_dict())
        target_model = target_model.eval()
        avg_rewards = []
        current_epoch = 0
        all_future_states = []
        corresponding_nums = []
        corresponding_nums_set = set()

        if stabilization:
            for i in range(min(128,2**N)):
                rand_num = random.randint(0,2**N-1)
                while rand_num in corresponding_nums_set:
                    rand_num = random.randint(0,2**N-1)
                corresponding_nums_set.add(rand_num)
                next_state = [int(j) for j in bin(rand_num)[2:].zfill(N)]
                all_future_states.append(next_state)
                corresponding_nums.append(rand_num)

    else:
        past_states, past_actions, past_rewards, past_probs, q_losses, q_network, criterion, optimizer, target_model, current_epoch, all_future_states, corresponding_nums, avg_rewards = memory
    

    current_epoch += 1

    if current_epoch < train_epochs:
         if len(past_states) > batch_size+2:
            start = time.time()
            random_memories = random.sample(list(range(len(past_states)-2)),batch_size)
            sample_state = np.array(past_states)[random_memories]
            sample_action = np.array(past_actions)[random_memories]
            sample_probs = np.array(past_probs)[random_memories]
            action_idx = np.array([action_to_idx[''.join([str(j) for j in i])] for i in sample_action])
            action_idx = torch.Tensor(action_idx).long()
            sample_reward = np.array(past_rewards)[random_memories]
            sample_reward = torch.Tensor(sample_reward)
            sample_state_next = np.array(past_states)[np.array(random_memories)+1]

            current_state_q = q_network(torch.Tensor(torch.Tensor(sample_state)))
            action_idx = action_idx.unsqueeze(1)
            current_state_value = torch.gather(current_state_q, 1, action_idx)

            if not stabilization:
                future_state_q = target_model(torch.Tensor(sample_state_next))
                future_state_value = future_state_q.max(1).values
                total_current_value = future_state_value * discount + sample_reward 
            else:
                future_state_q = target_model(torch.Tensor(torch.Tensor(all_future_states)))
                future_state_value = future_state_q.max(1).values.float()

                prob_list = torch.Tensor(sample_probs)
                discounted_future_values = torch.matmul(future_state_value, discount * prob_list.T.float())
                future_values = discounted_future_values / torch.sum(prob_list, dim=1)

                total_current_value = sample_reward + future_values

            loss = criterion(current_state_value.squeeze(),total_current_value)
            q_losses.append(loss.item())

            avg_rewards.append(q_network(torch.Tensor([[1 for i in range(len(state))]]))[0][1].item())

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 


    # Update the target model
    if current_epoch < train_epochs and current_epoch % target_update_freq == 0:
        for target_param, local_param in zip(target_model.parameters(), q_network.parameters()):
                target_param.data.copy_(tau*local_param.data+(1-tau)*target_param.data)

    
    # # Compute the best action
    action_values = q_network(torch.Tensor([state]))[0]

    max_action = valid_actions[torch.argmax(action_values)]

    if random.random() < epsilon and current_epoch < train_epochs:
        action = random.sample(valid_actions,1)[0]
    else:
        action = max_action
    

    past_states.append(state)
    past_actions.append(action)
    past_rewards.append(get_reward(state,action,match_probs,lamb))

    temp_list = []
    if stabilization:
        for idx in range(len(corresponding_nums)):
            next_state = all_future_states[idx]
            prob = 1 
            for j in range(N):
                prob *= env.transitions[j//env.volunteers_per_arm, state[j], action[j], next_state[j]]
            temp_list.append(prob)

    past_probs.append(temp_list)


    if len(past_states) > MAX_REPLAY_SIZE:
        past_states = past_states[-MAX_REPLAY_SIZE:]
        past_actions = past_actions[-MAX_REPLAY_SIZE:]
        past_rewards = past_rewards[-MAX_REPLAY_SIZE:]
        past_probs = past_probs[-MAX_REPLAY_SIZE:]

    # Repack the memory
    memory = past_states, past_actions, past_rewards, past_probs, q_losses, q_network, criterion, optimizer, target_model, current_epoch, all_future_states, corresponding_nums, avg_rewards

    return action, memory

def dqn_stable_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none"):
    return dqn_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",stabilization=True)

def dqn_with_stablization_steps(env,state,budget,lamb,memory,per_epoch_results,group_setup="none"):
    return dqn_with_steps(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",stabilization=True)

def dqn_max_with_stablization_steps(env,state,budget,lamb,memory,per_epoch_results,group_setup="none"):
    return dqn_with_steps(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",stabilization=True,use_max=True)

def dqn_max_with_steps(env,state,budget,lamb,memory,per_epoch_results,group_setup="none"):
    return dqn_with_steps(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",use_max=True)

def dqn_with_steps(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",stabilization=False,use_max=False):
    """Use a DQN policy to compute the action values
    
    Arguments: 
        env: Simulator Environment
        state: Num Agents x 2 numpy array (0-1)
        budget: Integer, how many arms we can pull
        Lamb: Balance between engagement, global reward
        Memory: Contains the V, Pi network
        per_epoch_results: Optional argument, nothing for this 
    Returns: Numpy array, action"""        
    # Hyperparameters + Parameters 
    
    value_lr = env.value_lr 
    train_epochs = env.train_epochs
    N = len(state) 
    match_probs = np.array(env.match_probability_list)[env.agent_idx]
    epsilon = 0.01
    target_update_freq = 1
    batch_size = 16
    discount = env.discount
    MAX_REPLAY_SIZE = 512
    tau = 0.01
    
    # Unpack the memory 
    if memory == None:
        past_states = []
        past_actions = []
        past_final_actions = []
        past_rewards = []
        past_probs = []
        past_next_states = []
        q_losses = []
        q_network = MLP(len(state)*2, 128, len(state))
        q_network.fc3.bias = torch.nn.Parameter(torch.Tensor([round(env.avg_reward) for i in range(len(q_network.fc3.bias))]))
        criterion = nn.MSELoss()
        optimizer = optim.Adam(q_network.parameters(), lr=value_lr)
        target_model = MLP(len(state)*2, 128, len(state))
        target_model.load_state_dict(q_network.state_dict())
        target_model = target_model.eval()
        avg_rewards = []
        current_epoch = 0

        all_future_states = []
        corresponding_nums = []
        corresponding_nums_set = set()
        if stabilization: 
            for i in range(min(env.num_samples,2**N)):
                rand_num = random.randint(0,2**N-1)
                while rand_num in corresponding_nums_set:
                    rand_num = random.randint(0,2**N-1)
                corresponding_nums_set.add(rand_num)
                next_state = [int(j) for j in bin(rand_num)[2:].zfill(N)]
                all_future_states.append(next_state + [0 for i in range(N)])
                corresponding_nums.append(rand_num)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.01)


    else:
        past_states, past_actions, past_final_actions, past_rewards, past_probs, past_next_states, q_losses, q_network, criterion, optimizer, target_model, current_epoch, all_future_states, corresponding_nums,scheduler,  avg_rewards = memory
    
    while len(past_states) > len(past_next_states):
        past_next_states.append(list(state)+[0 for i in range(len(state))])

    current_epoch += 1

    if current_epoch < train_epochs:
         if len(past_states) > batch_size+2:
            start = time.time() 
            random_memories = random.sample(list(range(len(past_states)-2)),batch_size)
            sample_state = np.array(past_states)[random_memories]
            sample_action = np.array(past_actions)[random_memories]
            sample_final_action = np.array(past_final_actions)[random_memories]
            action_idx = torch.Tensor(sample_action).long().unsqueeze(1) 
            sample_reward = np.array(past_rewards)[random_memories]
            sample_probs = np.array(past_probs)[random_memories]

            sample_reward = torch.Tensor(sample_reward)
            sample_state_next = np.array(past_next_states)[np.array(random_memories)]

            current_state_q = q_network(torch.Tensor(torch.Tensor(sample_state)))
            current_state_value = torch.gather(current_state_q, 1, action_idx)
            
            if not stabilization:
                future_state_q = target_model(torch.Tensor(sample_state_next))
                future_state_value = future_state_q.max(1).values
                total_current_value = future_state_value * discount + sample_reward 
            else:
                total_current_value = sample_reward
                future_values = torch.zeros(len(sample_state))
                future_state_q = target_model(torch.Tensor(torch.Tensor(all_future_states)))
                
                future_state_value = future_state_q.max(1).values                 
                prob_list = torch.Tensor(sample_probs)
                discounted_future_values = torch.matmul(future_state_value, discount * prob_list.T.float())
                future_values += discounted_future_values / torch.sum(prob_list, dim=1)

                total_current_value += future_values

            loss = criterion(current_state_value.squeeze(),total_current_value)
            q_losses.append(loss.item())
            avg_rewards.append(q_network(torch.Tensor([[1 for i in range(len(state))] + [0 for i in range(len(state))]]))[0].detach().numpy())

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            scheduler.step()

    start = time.time() 
    # Update the target model
    if current_epoch < train_epochs and current_epoch % target_update_freq == 0:
        for target_param, local_param in zip(target_model.parameters(), q_network.parameters()):
                target_param.data.copy_(tau*local_param.data+(1-tau)*target_param.data)
    target_model = target_model.eval() 
    
    # # Compute the best action
    final_action = [0 for i in range(len(state))]
    state = list(state)

    for _ in range(budget):
        action_values = q_network(torch.Tensor([state+final_action]))[0]
        played_actions = [i for i in range(len(final_action)) if final_action[i] == 1]
        unplayed_actions = [i for i in range(len(final_action)) if final_action[i] == 0]
        for j in played_actions:
            action_values[j] = float("-inf")
        max_action = torch.argmax(action_values)

        if random.random() < epsilon and current_epoch < train_epochs:
            action = random.sample(unplayed_actions,1)[0]
        else:
            action = max_action.item()

        past_states.append(state + final_action)
        past_actions.append(action)

        final_action[action] = 1
    
    state = np.array(state)
    action = np.array(final_action)

    for _ in range(budget):
        past_final_actions.append(final_action)

    if use_max:
        rew = get_reward_max(state,action,match_probs,lamb)
    else:
        rew = get_reward(state,action,match_probs,lamb)

    for i in range(budget):
        past_rewards.append(rew)

    temp_list = []
    if stabilization: 
        for idx in range(len(corresponding_nums)):
            next_state = all_future_states[idx]
            prob = 1 
            for j in range(N):
                prob *= env.transitions[j//env.volunteers_per_arm, state[j], action[j], next_state[j]]
            temp_list.append(prob)
    
    for _ in range(budget):
        past_probs.append(temp_list)


    if len(past_states) > MAX_REPLAY_SIZE:
        past_states = past_states[-MAX_REPLAY_SIZE:]
        past_actions = past_actions[-MAX_REPLAY_SIZE:]
        past_final_actions = past_final_actions[-MAX_REPLAY_SIZE:]
        past_rewards = past_rewards[-MAX_REPLAY_SIZE:]
        past_probs = past_probs[-MAX_REPLAY_SIZE:]

    # Repack the memory
    memory = past_states, past_actions, past_final_actions, past_rewards, past_probs, past_next_states, q_losses, q_network, criterion, optimizer, target_model, current_epoch, all_future_states, corresponding_nums,scheduler,  avg_rewards

    return action, memory

class MonteCarloTreeSearchNode():
    def __init__(self, state, simulation_no,transitions,parent=None, parent_action=None,use_whittle=False,memoizer=Memoizer('optimal')):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = []
        self.results_children = {}
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.transitions = transitions 
        self.use_whittle = use_whittle
        self.memoizer = memoizer
        self.simulation_no = simulation_no
        return

    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):
        if self.is_terminal_node():
            return np.mean(self._results)*self._number_of_visits
        else:
            return np.max(list(self.results_children.values()))*self._number_of_visits

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = StateAction.move(self.state,action,self.transitions)
        next_state.memory = self.state.memory 
        child_node = MonteCarloTreeSearchNode(
            next_state, self.simulation_no,self.transitions,parent=self, parent_action=action,use_whittle=self.use_whittle)

        self.children.append(child_node)
        return child_node 

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        
        tick = 0
        while not current_rollout_state.is_game_over():
            tick += 1
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action,self.transitions)        

        return current_rollout_state.game_result()

    def backpropagate(self, result,action=None):
        self._number_of_visits += 1.
        if self.is_terminal_node():
            self._results.append(result)
            result = np.mean(self._results)
        else:
            self.results_children[action] = result 
        if self.parent:
            self.parent.backpropagate(result,self.parent_action)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=5):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):  
        if self.use_whittle:
            state_WI = whittle_index(self.state.env,self.state.current_state,self.state.budget,self.state.lamb,self.memoizer)
            relevant_WI = [state_WI[i] for i in possible_moves]
            best_move = possible_moves[np.argmax(relevant_WI)]
            return best_move 
        return possible_moves[random.randint(0,len(possible_moves)-1)]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self,budget):
        simulation_no = self.simulation_no
        for i in range(simulation_no):
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        curr_node = self 
        actions = []
        for i in range(budget):
            if len(curr_node.children) == 0:
                possible_values = curr_node.state.get_legal_actions()
                actions += random.sample(possible_values,budget-i)
            else:
                best_child = curr_node.best_child(c_param=0.)
                actions.append(best_child.parent_action)
                curr_node = best_child 
        return actions 
    
    def __str__(self):
        return str(self.state)

class StateAction():
    def __init__(self,budget,discount,lamb,initial_state,volunteers_per_arm,n_arms,match_probs,max_rollout_actions,env,shapley=False):
        self.budget = budget 
        self.discount = discount 
        self.lamb = lamb  
        self.volunteers_per_arm = volunteers_per_arm 
        self.n_arms = n_arms
        self.match_probs = match_probs 
        self.max_rollout_actions = max_rollout_actions 
        self.previous_state_actions = []
        self.current_state = initial_state 
        self.env = env 
        self.shapley = shapley 
        self.memory = None
        self.attribution_method = "proportional"

    def get_legal_actions(self):
        idx = len(self.previous_state_actions)//self.budget*self.budget 
        current_state_actions = self.previous_state_actions[idx:]
        taken_actions = set([i[1] for i in current_state_actions])
        all_actions = set(range(self.volunteers_per_arm * self.n_arms))
        valid_actions = list(all_actions.difference(taken_actions))

        return valid_actions 
    
    def is_game_over(self):
        return len(self.previous_state_actions) >= self.max_rollout_actions 
    
    def game_result(self):
        total_reward = 0
        last_state = []
        last_action = []
        base_state = self.previous_state_actions[0][0]
        state_WI = whittle_index(self.env,base_state,self.budget,self.lamb,self.memory[0],reward_function="combined",match_probs=self.memory[1])
        sorted_WI = np.argsort(state_WI)[::-1]
        base_action = np.zeros(len(state_WI), dtype=np.int8)
        base_action[sorted_WI[:self.budget]] = 1
        base_reward = get_reward_custom(base_state,base_action,self.match_probs,self.lamb)
        base_WI = np.sum(state_WI[base_action == 1])

        for i in range(self.max_rollout_actions//self.budget):
            state_choices = self.previous_state_actions[i*self.budget:(i+1)*self.budget]
            corresponding_actions = [i[1] for i in state_choices]
            corresponding_state = np.array(state_choices[0][0])
            action_0_1 = []
            for arm in range(len(corresponding_state)):
                if arm in corresponding_actions:
                    action_0_1.append(1)
                else:
                    action_0_1.append(0)
            action_0_1 = np.array(action_0_1)
            total_reward += self.discount**i * get_reward_custom(corresponding_state,action_0_1,self.match_probs,self.lamb)
            last_state = corresponding_state
            last_action = action_0_1

        if self.shapley:
            assert self.max_rollout_actions == self.budget 
            memory_whittle, memory_shapley = self.memory 
            match_prob_now = get_attributions(last_state,last_action,self.match_probs,self.lamb,memory_shapley,attribution_method=self.attribution_method)
            state_WI = whittle_index(self.env,last_state,self.budget,self.lamb,memory_whittle,reward_function="combined",match_probs=memory_shapley,match_prob_now=match_prob_now)

            total_reward = np.sum(np.array(state_WI)[last_action == 1])
            # state_WI = []
            # for i in range(len(last_state)):
            #     if last_state[i] == 1:
            #         state_WI.append(arm_compute_whittle_multi_prob(self.env.transitions[i//self.env.volunteers_per_arm,:,:,1], last_state[i], self.discount, 100, eps=1e-3,reward_function='combined',lamb=self.lamb,match_prob=memory_shapley[i],match_prob_now=memory_shapley_normalized[i],num_arms=len(last_state))) 
            #     else:
            #         state_WI.append(
            #             arm_compute_whittle(self.env.transitions[i//self.env.volunteers_per_arm,:,:,1], last_state[i], self.discount, 100, eps=1e-3,reward_function='combined',lamb=self.lamb,match_prob=memory_shapley[i],num_arms=len(last_state)))
            # total_reward_2 = np.sum(np.array(state_WI)[last_action == 1])
            # assert abs(total_reward_2-total_reward) < 0.001

            # memory_whittle, memory_shapley = self.memory 
            # state_WI = whittle_index(self.env,last_state,self.budget,self.lamb,memory_whittle,reward_function="combined",match_probs=memory_shapley)
            # self.memory = (memory_whittle, memory_shapley)
            # score_diff = (total_reward-(np.sum((1-self.lamb)*last_state*last_action*memory_shapley)+np.sum(last_state)/len(last_state)*self.lamb))
            # memory_shapley_normalized = deepcopy(memory_shapley)
            # memory_shapley_normalized *= last_state*last_action 
            # if sum(memory_shapley_normalized) > 0:
            #     memory_shapley_normalized /= np.sum(memory_shapley_normalized)
            # state_WI += memory_shapley_normalized * score_diff 


            # total_reward = np.sum(state_WI[last_action == 1]) 
        return total_reward
    
    def move(initial_state,action,transitions):
        previous_state_actions = initial_state.previous_state_actions + [(initial_state.current_state,action)]

        new_state_object = StateAction(initial_state.budget,initial_state.discount,initial_state.lamb,initial_state.current_state,initial_state.volunteers_per_arm,initial_state.n_arms,initial_state.match_probs,initial_state.max_rollout_actions,initial_state.env)
        new_state_object.attribution_method = initial_state.attribution_method
        new_state_object.previous_state_actions = previous_state_actions
        new_state_object.shapley = initial_state.shapley 
        new_state_object.memory = initial_state.memory 
        return new_state_object 
    
    def __str__(self):
        return str(self.previous_state_actions)


def mcts_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none"):
    N = len(state)
    rollout = budget
    match_probs = np.array(env.match_probability_list)[env.agent_idx]
    state_actions = []
    if memory == None:
        memory = Memoizer('optimal'),np.array(shapley_index_custom(env,np.ones(len(env.agent_idx)),{})[0])

    for i in range(budget):
        s = StateAction(budget,env.discount,lamb,state,env.volunteers_per_arm,env.cohort_size,match_probs,rollout,env)
        s.previous_state_actions = state_actions
        s.memory = memory 
        root = MonteCarloTreeSearchNode(s,env.mcts_test_iterations,transitions=env.transitions)
        selected_node = root.best_action()
        state_actions = selected_node.state.previous_state_actions
        memory = s.memory 

    selected_idx = [i[1] for i in state_actions]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, memory 

def get_attributions(state,action,match_probs,lamb,shapley_indices,attribution_method="proportional"):
    """Detemrine how much reward to give to each action
    
    Arguments:  
        state: Numpy array of length N, 0-1
        action: Numpy array of length N, 0-1
        match_probs: Marginal values for each arm
        lamb: Float, balance between matching, engagement
        shapley_indices: Computed impact of each arm"""

    if attribution_method == "proportional":
        reward = get_reward_custom(state,action,match_probs,lamb)
        memory_shapley_normalized = shapley_indices*state*action 
        if np.sum(memory_shapley_normalized) > 0:
            memory_shapley_normalized /= np.sum(memory_shapley_normalized)
        memory_shapley_normalized *= reward  
        return memory_shapley_normalized
    elif attribution_method == "shapley":
        shapley_values = []
        for i in range(len(action)):
            if state[i]*action[i] == 0:
                shapley_values.append(0)
            else:
                action_copy = deepcopy(action)
                action_copy[i] = 0
                reward_difference = get_reward_custom(state,action,match_probs,lamb)
                reward_difference -= get_reward_custom(state,action_copy,match_probs,lamb)
                shapley_values.append(reward_difference)
        return np.array(shapley_values)
    else:
        raise Exception("Method {} not found".format(attribution_method))

def mcts_shapley_attributions_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none"):
    return mcts_shapley_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",attribution_method="shapley")

def mcts_shapley_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",attribution_method="proportional"):
    N = len(state)
    rollout = budget
    match_probs = np.array(env.match_probability_list)[env.agent_idx]
    state_actions = []

    if memory == None:
        memory = Memoizer('optimal'),np.array(shapley_index_custom(env,np.ones(len(env.agent_idx)),{})[0])

    s = StateAction(budget,env.discount,lamb,state,env.volunteers_per_arm,env.cohort_size,match_probs,rollout,env,shapley=True)
    s.attribution_method = attribution_method 
    s.previous_state_actions = state_actions
    s.memory = memory 
    s.per_epoch_results = per_epoch_results
    root = MonteCarloTreeSearchNode(s,env.mcts_test_iterations,transitions=env.transitions)
    selected_idx = root.best_action(budget)
    memory = s.memory 

    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1
    match_prob_now = get_attributions(state,action,match_probs,lamb,memory[1],attribution_method=attribution_method)
    computed_whittle = whittle_index(env,state,budget,lamb,memory[0],reward_function="combined",match_probs=memory[1],match_prob_now=match_prob_now)
    action_index =  np.sum(computed_whittle[action == 1])

    shapley_action = shapley_whittle_custom_policy(env,state,budget,lamb,memory, per_epoch_results)[0]
    match_prob_now = get_attributions(state,shapley_action,match_probs,lamb,memory[1],attribution_method=attribution_method)
    computed_whittle = whittle_index(env,state,budget,lamb,memory[0],reward_function="combined",match_probs=memory[1],match_prob_now=match_prob_now)
    shapley_action_index = np.sum(computed_whittle[shapley_action == 1])

    if shapley_action_index > action_index:
        action = shapley_action 

    return action, memory


def mcts_whittle_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none"):
    N = len(state)
    memoizer = memory 
    if memoizer == None:
        memozier = Memoizer("optimal")

    rollout = budget*5
    match_probs = np.array(env.match_probability_list)[env.agent_idx]
    state_actions = []

    for i in range(budget):
        s = StateAction(budget,env.discount,lamb,state,env.volunteers_per_arm,env.cohort_size,match_probs,rollout,env)
        s.previous_state_actions = state_actions
        root = MonteCarloTreeSearchNode(s,env.mcts_test_iterations,transitions=env.transitions,use_whittle=True,memoizer=memoizer)
        selected_node = root.best_action()
        state_actions = selected_node.state.previous_state_actions

    selected_idx = [i[1] for i in state_actions]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, memory 


def full_mcts_policy(env,state,budget,lamb,memory,per_epoch_results,contextual=False,group_setup="transition",
                        run_ucb = True, use_whittle=True):
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
        criterion_policy = nn.BCEWithLogitsLoss()
        memoizer = Memoizer('optimal')
        policy_network_size = 6+len(state)
        if contextual:
            policy_network_size = 6+len(state) 
        policy_network = MLP(policy_network_size,512,1)
        optimizer_policy = optim.Adam(policy_network.parameters(),lr=policy_lr)

        best_group_arms = [] 
        if contextual:
            match_probabilities = env.current_episode_match_probs[:,env.agent_idx]
            match_probs = np.mean(match_probabilities,axis=0)
        else:
            match_probs = [] 
            whittle_index(env,[0 for i in range(len(all_match_probs))],budget,lamb,memoizer)
            whittle_index(env,[1 for i in range(len(all_match_probs))],budget,lamb,memoizer)
    else:
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
    best_group_arms = get_groups(group_setup,best_group_arms,state,env,match_probs,all_match_probs,contextual,lamb,memoizer,budget) 
    num_groups = len(best_group_arms)


    # Step 2: MCTS iterations
    # Get the probability for pulling each arm, in the same order
    # As the 'best_group_arms' variable
    action = get_action_state(policy_network,[state],env,contextual=contextual,memoizer=memoizer)
    x_points = get_policy_network_input_many_state(env,np.array([state]),contextual=contextual,memoizer=memoizer)

    policy_network_predictions = np.array(policy_network(torch.Tensor(x_points)).detach().numpy().T)[0]
    policy_by_group = []
    for g in range(num_groups):
        policy_by_group.append(policy_network_predictions[best_group_arms[g]])

    action = np.zeros(len(state), dtype=np.int8)
    action[np.argsort(policy_network_predictions)[::-1][:budget]] = 1
    full_combo, group_indices = action_to_group_combo(action,best_group_arms)
    total_value = get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,policy_network,lamb,num_future_samples=25,contextual=contextual,memoizer=memoizer)
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

    if use_whittle: 
        action = np.zeros(len(state), dtype=np.int8)
        action[np.argsort(state_WI)[::-1][:budget]] = 1
        full_combo, group_indices = action_to_group_combo(action,best_group_arms)
        total_value = get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,policy_network,lamb,num_future_samples=25,contextual=contextual,memoizer=memoizer)
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
    
            if run_ucb: 
            # Find upper bounds to determine if it's worthwhile to keep exploring 
                UCB_by_arm = {}
                should_random = False 
                should_break = False 
                if len(scores_current_combo) > 0:
                    current_value = get_total_value(env,all_match_probs,best_group_arms,state,
                                                    group_indices,value_network,policy_network,lamb,num_future_samples=25,contextual=contextual,memoizer=memoizer)
                    for arm in scores_current_combo:
                        new_group_index = deepcopy(group_indices)
                        new_group_index[arm] += 1
                        value_with_pull = get_total_value(env,all_match_probs,best_group_arms,state,new_group_index,value_network,policy_network,lamb,num_future_samples=10,contextual=contextual,memoizer=memoizer)
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
            else:
                should_random = False 
                UCB_by_arm = {}
                for arm in scores_current_combo:
                    UCB_by_arm[arm] = np.mean(scores_current_combo[arm])

            # Determine which arm to explore
            if not should_random:
                should_random = random.random() <= epsilon_func(n)

            if should_random:
                selected_index, memoizer = get_random_index(env,all_match_probs,best_group_arms,policy_by_group,group_indices,num_epochs,lamb,memoizer,state,budget,state_WI,randomness_boost,use_whittle=use_whittle)
            else:
                selected_index = max(UCB_by_arm, key=UCB_by_arm.get)
            update_combos.append((current_combo[:],selected_index))
            current_combo.append(selected_index)
            group_indices[selected_index] += 1 

        if skip_iteration:
            continue 
    
        should_break = False 

        # Get the total value, and update this along subsets of arms pulled 
        total_value = get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,policy_network,lamb,num_future_samples=25,contextual=contextual,memoizer=memoizer)
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
    best_value = get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,policy_network,lamb,num_future_samples=25,contextual=contextual,memoizer=memoizer)
    action = np.zeros(len(state), dtype=np.int8)
    action[combo_to_volunteers] = 1

    # Update datasets + backprop 
    past_states.append(list(state))
    past_values.append(best_value)
    past_actions.append(action)

    if num_epochs < train_epochs and (num_epochs+1)%network_update_frequency == 0:
        for i in range(num_network_updates):
            update_value_function(past_states,past_actions,value_losses,value_network,criterion_value,optimizer_value,
                                    policy_network,env,lamb,contextual=contextual,memoizer=memoizer)

            update_policy_function(past_states,past_actions,policy_losses,policy_network,
            criterion_policy,optimizer_policy,env,contextual=contextual,memoizer=memoizer)
    if contextual:
        memory = past_states, past_values, past_actions, value_losses, value_network, criterion_value, optimizer_value, policy_losses, policy_network, criterion_policy, optimizer_policy, memoizer, {'match_probs': match_probs, 'best_group_arms': best_group_arms} 
    else:
        memory = past_states, past_values, past_actions, value_losses, value_network, criterion_value, optimizer_value, policy_losses, policy_network, criterion_policy, optimizer_policy, memoizer, {'match_probs': [], 'best_group_arms': best_group_arms} 

    return action, memory 