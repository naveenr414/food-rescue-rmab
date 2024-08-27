from rmab.mcts_policies import get_reward_custom
import numpy as np
import random 

from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
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


def dqn_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",stabilization=False,greedy_eval=False):
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


    if greedy_eval:
        curr_action = [0 for i in range(len(state))]

        for i in range(budget):
            best_action = 0 
            best_reward = -1 
            for j in range(len(state)):
                if curr_action[j] == 0:
                    new_action = deepcopy(curr_action)
                    new_action[j] = 1
                    idx = action_to_idx[''.join([str(k) for k in new_action])]
                    reward = action_values[idx]

                    if reward > best_reward:
                        best_reward = reward 
                        best_action = j 
            curr_action[best_action] = 1
        max_action = np.array(curr_action)
    else: 
        max_action = valid_actions[torch.argmax(action_values)]


    if random.random() < epsilon and current_epoch < train_epochs:
        action = random.sample(valid_actions,1)[0]
    else:
        action = max_action
    

    past_states.append(state)
    past_actions.append(action)
    rew = get_reward_custom(state,action,match_probs,lamb,env.reward_type,env.reward_parameters,env.context)

    past_rewards.append(rew)

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
    
    value_lr = 1e-4 
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

    rew = get_reward_custom(state,action,match_probs,lamb,env.reward_type,env.reward_parameters,env.context)

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

def dqn_policy_greedy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none"):
    """Use a DQN policy + greedily select arms 
    
    Arguments: 
        env: Simulator Environment
        state: Num Agents x 2 numpy array (0-1)
        budget: Integer, how many arms we can pull
        Lamb: Balance between engagement, global reward
        Memory: Contains the V, Pi network
        per_epoch_results: Optional argument, nothing for this 
    Returns: Numpy array, action"""        

    return dqn_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",greedy_eval=True)

def dqn_stable_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none"):
    """Use a DQN + Stabilize the policy by averaging over different combinations of actions
    
    Arguments: 
        env: Simulator Environment
        state: Num Agents x 2 numpy array (0-1)
        budget: Integer, how many arms we can pull
        Lamb: Balance between engagement, global reward
        Memory: Contains the V, Pi network
        per_epoch_results: Optional argument, nothing for this 
    Returns: Numpy array, action"""        

    return dqn_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",stabilization=True)

def dqn_with_stablization_steps(env,state,budget,lamb,memory,per_epoch_results,group_setup="none"):
    """Use a DQN policy while letting the actions be one arm, and stabilize this
    
    Arguments: 
        env: Simulator Environment
        state: Num Agents x 2 numpy array (0-1)
        budget: Integer, how many arms we can pull
        Lamb: Balance between engagement, global reward
        Memory: Contains the V, Pi network
        per_epoch_results: Optional argument, nothing for this 
    Returns: Numpy array, action"""        

    return dqn_with_steps(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",stabilization=True)