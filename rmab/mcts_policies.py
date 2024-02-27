import numpy as np
import random 

from rmab.uc_whittle import Memoizer
from rmab.omniscient_policies import whittle_greedy_policy, whittle_index

from copy import deepcopy
from mcts import mcts
import time 

import torch
import torch.nn as nn
import torch.optim as optim


class VolunteerState():
    def __init__(self):
        self.non_match_prob = 1
        self.previous_pulls = set()
        self.N = 0
        self.arm_values = []
        self.budget = 0
        self.states = []
        self.discount = 0.9
        self.lamb = 0
        self.max_depth = 1
        self.rollout_number = 1
        self.previous_rewards = []
        self.transitions = []
        self.cohort_size = 0
        self.volunteers_per_arm = 0
        self.memory = {}
        self.memoizer = Memoizer('optimal')

    def getPossibleActions(self):
        possibleActions = []
        for i in range(self.N):
            if i not in self.previous_pulls:
                possibleActions.append(Action(arm=i))
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.previous_pulls.add(action.arm)
        newState.memoizer = self.memoizer

        return newState

    def isTerminal(self):
        if len(self.previous_pulls) == self.budget:
            return True 
        return False         

    def getReward(self):
        """Compute the reward using the Whittle index + Matching
            For each combo"""

        policy = 'index'

        if policy == 'reward':
            non_match_prob = 1 
            for i in self.previous_pulls:
                expected_active_rate = 0
                for i in range(len(self.states)):
                    action = int(i in self.previous_pulls)
                    expected_active_rate += self.transitions[i//self.volunteers_per_arm,self.states[i],action,1]

                non_match_prob *= (1-self.states[i]*self.arm_values[i])

            value = 1-non_match_prob  + self.lamb*expected_active_rate

        elif policy == 'index':
            state_WI = whittle_index(self.env,self.states,self.budget,self.lamb,self.memoizer,reward_function="activity")
            non_match_prob = 1 

            for i in self.previous_pulls:
                non_match_prob *= (1-self.env.transitions[i//self.volunteers_per_arm,self.states[i],1,1]*self.arm_values[i])

            value = (1-non_match_prob)/(1-self.discount) + self.lamb*np.sum(state_WI[list(self.previous_pulls)]) 

        return value 

    def set_state(self,states):
        self.states = states

class VolunteerStateTwoStepMCTS(VolunteerState):
    def __init__(self):
        super().__init__()

        self.num_by_cohort = []
        self.current_cohort = 0
        self.num_in_cohort = 0

    def set_num_by_cohort(self,num_by_cohort):
        """Each level in the MCTS search is a different group  
            that we aim to look at
            The current cohort is the current group we're looking at"""

        self.num_by_cohort = num_by_cohort
        self.current_cohort = 0
        while self.num_by_cohort[self.current_cohort] == 0:
            self.current_cohort += 1

    def getPossibleActions(self):
        possibleActions = []
        for i in range(self.volunteers_per_arm):
            idx = self.current_cohort*self.volunteers_per_arm + i
            if idx not in self.previous_pulls:
                possibleActions.append(Action(arm=idx))
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.previous_pulls.add(action.arm)
        newState.memoizer = self.memoizer
        newState.num_in_cohort += 1
        if newState.num_in_cohort >= newState.num_by_cohort[newState.current_cohort]:
            newState.num_in_cohort = 0
            newState.current_cohort += 1
            while newState.current_cohort < len(
                newState.num_by_cohort) and newState.num_by_cohort[newState.current_cohort] == 0:
                newState.current_cohort += 1

        return newState


class Action():
    def __init__(self, arm):
        self.arm = arm 

    def __str__(self):
        return str(self.arm)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.arm == other.arm

    def __hash__(self):
        return hash((self.arm))

def set_initial_state(initialState,env,state,memory,lamb):
    """Set the hyperparameter values for the initial state
        based on environment + state information
        
    Arguments:
        initialState: VolunterState or VolunterStateTwoStep
        env: Simulator Environmenet
        state: Numpy array of the current state
        memory: List with the first element being a dictionary
            The second being a memoizer
        lamb: Float, the current \lambda value
            
    Returns: The initialState, with all the parameters set"""

    initialState.arm_values = np.array(env.match_probability_list)[env.agent_idx] 
    initialState.N = len(initialState.arm_values)
    initialState.budget = env.budget 
    initialState.transitions = env.transitions
    initialState.cohort_size = env.cohort_size
    initialState.volunteers_per_arm = env.volunteers_per_arm
    initialState.discount = env.discount 
    initialState.env = env 
    initialState.lamb = lamb
    initialState.set_state(state)
    initialState.memory = memory[0]
    initialState.memoizer = memory[1]

    return initialState

def one_step_idx_to_action(selected_idx,env,state,lamb,memory):
    """Turn a list of volunteers to notify into a 0-1 
        vector for action
        
    Arguments:
        selected_idx: Numpy array of volunteers to notify
        env: RMAB Simulator Environment
        state: 0-1 numpy array for the state
        lamb: Float, what \lambda value
        memory: List with the first element as a dictionary
            second element as a memoizer
            
    Returns: 0-1 numpy array with the actions to take"""

    selected_idx = np.array(selected_idx)
    action = np.zeros(len(state), dtype=np.int8)
    action[selected_idx] = 1

    return action, memory

def two_step_mcts_to_action(env,state,lamb, memory): 
    """Run MCTS to find the groups, then run MCTS again to find
        which volunteers in each group to notify
        
    Arguments:
        selected_idx: Numpy array of volunteers to notify
        env: RMAB Simulator Environment
        state: 0-1 numpy array for the state
        lamb: Float, what \lambda value
        memory: List with the first element as a dictionary
            second element as a memoizer
            
    Returns: 0-1 numpy array with the actions to take"""


    fractions = [1/3,1/2]
    fractions.append((1-sum(fractions))/2)
    greedy_action, _ = mcts_policy(env,state,env.budget,lamb,memory,None,timeLimit=env.TIME_PER_RUN * fractions[0])
    bin_counts = np.zeros(env.cohort_size, dtype=int)
    for i in range(env.cohort_size):
        bin_counts[i] = np.sum(greedy_action[i*env.volunteers_per_arm:(i+1)*env.volunteers_per_arm])
    num_by_cohort = bin_counts

    initialState = VolunteerStateTwoStepMCTS()
    initialState = set_initial_state(initialState,env,state,memory,lamb)
    initialState.set_num_by_cohort(num_by_cohort)
    
    searcher = mcts(timeLimit=env.TIME_PER_RUN * fractions[1]) 
    selected_idx = []
    current_state = initialState

    for _ in range(env.budget):
        action = searcher.search(initialState=current_state)
        current_state = current_state.takeAction(action)
        selected_idx.append(action.arm)
        searcher = mcts(timeLimit=env.TIME_PER_RUN * fractions[2])

    action = np.zeros(len(state), dtype=np.int8)
    action[selected_idx] = 1

    return action, memory

def two_step_mcts_to_whittle(env,state,lamb, memory): 
    """Run Whittle+Greedy to find the groups, then run MCTS again to find
        which volunteers in each group to notify
        
    Arguments:
        selected_idx: Numpy array of volunteers to notify
        env: RMAB Simulator Environment
        state: 0-1 numpy array for the state
        lamb: Float, what \lambda value
        memory: List with the first element as a dictionary
            second element as a memoizer
            
    Returns: 0-1 numpy array with the actions to take"""


    fractions = [1/2]
    fractions.append((1-sum(fractions))/2)
    greedy_action, _ = whittle_greedy_policy(env,state,env.budget,lamb,memory[1],None)
    bin_counts = np.zeros(env.cohort_size, dtype=int)
    for i in range(env.cohort_size):
        bin_counts[i] = np.sum(greedy_action[i*env.volunteers_per_arm:(i+1)*env.volunteers_per_arm])
    num_by_cohort = bin_counts

    initialState = VolunteerStateTwoStepMCTS()
    initialState = set_initial_state(initialState,env,state,memory,lamb)

    initialState.set_num_by_cohort(num_by_cohort)
    
    searcher = mcts(timeLimit=env.TIME_PER_RUN * fractions[0]) 
    selected_idx = []
    current_state = initialState 

    for _ in range(env.budget):
        action = searcher.search(initialState=current_state)
        current_state = current_state.takeAction(action)
        selected_idx.append(action.arm)
        searcher = mcts(timeLimit=env.TIME_PER_RUN * fractions[1])

    action = np.zeros(len(state), dtype=np.int8)
    action[selected_idx] = 1

    return action, memory

def init_memory(memory):
    """For MCTS process, we store two things in the memory
        a) A dictionary with past iterations which we've solved
        b) A memoizer for the Whittle process
    
    Arguments:
        memory: None or a list with a dictionary and a memoizer
    
    Returns: A list with a dictionary and a memoizer"""

    if memory == None:
        memory = [{}, Memoizer('optimal')]
    return memory 

def run_two_step(env,state,budget,lamb,memory,per_epoch_results,idx_to_action,timeLimit=-1):
    """Run a two-step policy using the idx_to_action function
    
    Arguments: 
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Information on previously computed Whittle indices, the memoizer
        per_epoch_results: Any information computed per epoch; unused here
        idx_to_action: Function such as two_step_mcts_to_whittle
        timeLimit: Maximum time, in miliseconds, for all MCTS calls
            By default this should be env.TIME_PER_RUN
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=
        List with a dictionary and a memoizer"""
    
    memory = init_memory(memory)
    if timeLimit == -1:
        timeLimit = env.TIME_PER_RUN 

    if ''.join([str(i) for i in state]) in memory[0]:
        return memory[0][''.join([str(i) for i in state])], memory

    action, memory = idx_to_action(env,state,lamb,memory)
    memory[0][''.join([str(i) for i in state])] = action 
    return action, memory

def run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts_function,timeLimit=-1):
    """Compute an MCTS-based policy which selects arms to notify 
    sequentially, then rolls out for Q=5 steps to predict reward

    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Information on previously computed Whittle indices, the memoizer
        per_epoch_results: Any information computed per epoch; unused here
        timeLimit: Total time for all MCTS calls 
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    memory = init_memory(memory)

    if ''.join([str(i) for i in state]) in memory[0]:
        return memory[0][''.join([str(i) for i in state])], memory
    
    if timeLimit == -1:
        timeLimit = env.TIME_PER_RUN 

    initialState = VolunteerState()
    initialState = set_initial_state(initialState,env,state,memory,lamb)

    fraction_first_budget = 4/5
    fraction_other = (1-fraction_first_budget)/(budget-1)

    searcher = mcts_function(timeLimit=timeLimit*fraction_first_budget)
    
    selected_idx = []
    current_state = initialState 

    for _ in range(budget):
        action = searcher.search(initialState=current_state)
        current_state = current_state.takeAction(action)
        selected_idx.append(action.arm)
        searcher = mcts_function(timeLimit=timeLimit*fraction_other)

    action, memory = one_step_idx_to_action(selected_idx,env,state,lamb,memory)
    memory[0][''.join([str(i) for i in state])] = action 
    return action, memory

def mcts_policy(env,state,budget,lamb,memory,per_epoch_results,num_iterations=100,timeLimit=-1):
    return run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts,timeLimit=timeLimit)

def mcts_mcts_policy(env,state,budget,lamb,memory,per_epoch_results):
    return run_two_step(env,state,budget,lamb,memory,per_epoch_results,two_step_mcts_to_action)

def mcts_whittle_policy(env,state,budget,lamb,memory,per_epoch_results):
    return run_two_step(env,state,budget,lamb,memory,per_epoch_results,two_step_mcts_to_whittle)

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

    return 1/(n+1)**.5

def get_random_index(env,all_match_probs,best_group_arms,policy_by_group,group_indices,num_epochs,lamb,memoizer,state,budget):
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
            state_WI = whittle_index(env,state,budget,lamb,memoizer)
            state_WI*=lamb 

            match_probabilities = np.array(env.match_probability_list)[env.agent_idx]*state

            state_WI += match_probabilities

            score_by_group.append(state_WI[best_group_arms[g][group_indices[g]]]+0.001)
            score_by_group_policy.append(max(policy_by_group[g][group_indices[g]],0.001))

    # TODO: Be more efficient here
    fraction_policy = 1 - epsilon_func(num_epochs)
    sum_group = sum(score_by_group)
    sum_policy = sum(score_by_group_policy)
    weighted_probabilities = [(1-fraction_policy)*score_by_group[i]/sum_group + fraction_policy*score_by_group_policy[i]/sum_policy for i in range(len(score_by_group_policy))]

    selected_index = random.choices(range(len(score_by_group)), weights=weighted_probabilities, k=1)[0]
    return selected_index, memoizer 

def get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,num_future_samples=25):
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
        non_match_prob *= np.prod(1-all_match_probs[best_group_arms[g][0:group_indices[g]]]*state[best_group_arms[g][0:group_indices[g]]])

    match_prob = 1-non_match_prob

    future_value = 0

    action = np.zeros(len(state), dtype=np.int8)
    probs = []

    for g in range(num_groups):
        for j in range(group_indices[g]):
            action[best_group_arms[g][j]] = 1

    for i in range(len(state)):
        probs.append(env.transitions[i//env.volunteers_per_arm, state[i], action[i], 1])

    # TODO: Speed this part up
    samples = np.random.random((num_future_samples,len(state))) < probs 
    samples = samples.astype(int)
    future_value = torch.mean(value_network(torch.Tensor(samples))).item()

    total_value = match_prob + future_value * env.discount 

    return total_value 

def update_value_policy_network(past_states,past_values,past_actions,value_network,criterion_value,optimizer_value,policy_network,criterion_policy,optimizer_policy,value_losses):
    """Run backprop on the Value and Policy Networks
    
    Arguments:
        past_states: Feature dataset; list of previous state seen
        past_values: For each past state seen, what was the computed value? 
        value_network: PyTorch model mapping states to float values
        criterion_value: MSE Loss, used for loss computation
        optimizer_value: SGD Optimizer for the value network
        policy_network: PyTorch model mapping states to action probabilities
        criterion_policy: BCE Loss, used for loss computation
        optimizer_policy: SGD Optimizer for the policy network
        value_losses: List of losses from the value network; used for debugging
    
    Returns: Nothing

    Side Effects: Runs backpropogation on value_network and policy_network
        using a random previously seen point
    
    """

    random_point = random.randint(max(0,len(past_states)-100),len(past_states)-1)
    input_data = torch.Tensor(past_states[random_point])
    target = torch.Tensor([past_values[random_point]])
    output = value_network(input_data)
    loss = criterion_value(output,target)

    optimizer_value.zero_grad()
    loss.backward()
    optimizer_value.step()

    random_point = random.randint(max(0,len(past_states)-100),len(past_states)-1)
    input_data = torch.Tensor(past_states[random_point])
    target = torch.Tensor(past_actions[random_point])
    output = policy_network(input_data)
    loss_policy = criterion_policy(output,target)

    optimizer_policy.zero_grad()  
    loss_policy.backward()  
    optimizer_policy.step()  
    value_losses.append(loss_policy.item())

def get_best_combo(past_prefixes,budget):
    """Using the data we've seen from the MCTS run, determine which combo of arms to pull
    
    Arguments:
        past_prefixes: Dictionary mapping arms pulled to a dictionary with 
            next arm pulled + value
        budget: Integer, total number of arms to pull
    
    Returns: List, best combination of arms to pull"""

    combo_list = [] 
    seen_combos = set()
    for prefix in past_prefixes:
        if len(eval(prefix)) == budget-1:
            value = max(past_prefixes[prefix].values())
            combo = [index for index in past_prefixes[prefix] if past_prefixes[prefix][index] == value][0]
            combo = eval(prefix) + [combo]
            combo = sorted(combo)

            if ''.join([str(i) for i in combo]) not in seen_combos:
                combo_list.append((combo,value))
                seen_combos.add(''.join([str(i) for i in combo]))

    sorted_combos = sorted(combo_list,key=lambda k: k[1])[-3:]
    return [i[0] for i in sorted_combos]


def full_mcts_policy(env,state,budget,lamb,memory,per_epoch_results,timeLimit=-1):
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

    # Initialize the memory 
    if memory == None:
        past_states = []
        past_values = []
        past_actions = []
        value_losses = []
        value_network = MLP(len(state), 64, 1)
        criterion_value = nn.MSELoss()
        optimizer_value = optim.SGD(value_network.parameters(), lr=0.01)
        policy_network = MLP(len(state),64,len(state),use_sigmoid=True)
        criterion_policy = nn.BCELoss()
        optimizer_policy = optim.SGD(policy_network.parameters(),lr=0.05)
        memoizer = Memoizer('optimal')
    else:
        past_states, past_values, past_actions, value_losses, value_network, criterion_value, optimizer_value, policy_network, criterion_policy, optimizer_policy, memoizer = memory
    num_epochs = len(past_states)
    num_iterations = 25
    past_prefixes = {}

    # Step 1: Group Setup
    num_groups = len(state)//env.volunteers_per_arm * 2 
    best_group_arms = []
    all_match_probs = np.array(env.match_probability_list)[env.agent_idx]
    for g in range(len(state)//env.volunteers_per_arm):
        matching_probs = all_match_probs[g*env.volunteers_per_arm:(g+1)*env.volunteers_per_arm]
        sorted_matching_probs = np.argsort(matching_probs)[::-1] + g*env.volunteers_per_arm
        sorted_matching_probs_0 = [i for i in sorted_matching_probs if state[i] == 0]
        sorted_matching_probs_1 = [i for i in sorted_matching_probs if state[i] == 1]
        best_group_arms.append(sorted_matching_probs_0)
        best_group_arms.append(sorted_matching_probs_1)

    # Step 2: MCTS iterations
    # Get the probability for pulling each arm, in the same order
    # As the 'best_group_arms' variable
    policy_network_predictions = policy_network(torch.Tensor([state])).detach().numpy()[0]
    policy_by_group = []
    for g in range(num_groups):
        policy_by_group.append(policy_network_predictions[best_group_arms[g]])

    if num_epochs > 2000:
        action = np.zeros(len(state), dtype=np.int8)
        action[np.argsort(policy_network_predictions)[::-1][:budget]] = 1
        return action, memory

    for _ in range(num_iterations):
        current_combo = [] 
        group_indices = [0 for i in range(num_groups)]
        update_combos = []
            
        for k in range(budget):
            if repr(current_combo) in past_prefixes:
                n = len(past_prefixes[repr(current_combo)])
            elif _ > 10 and k == 0:
                n = 1
            else:
                n = 0

            should_random = random.random() <= epsilon_func(n)

            if should_random:
                selected_index, memoizer = get_random_index(env,all_match_probs,best_group_arms,policy_by_group,group_indices,num_epochs,lamb,memoizer,state,budget)
            else:
                scores_current_combo = past_prefixes[repr(current_combo)]
                selected_index = max(scores_current_combo, key=scores_current_combo.get)

            # TODO: Can we be even more sample efficient here? Use all the combos? 
            update_combos.append((current_combo[:],selected_index))
            current_combo.append(selected_index)
            group_indices[selected_index] += 1 

        # Get the value 
        total_value = get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,num_future_samples=25)

        for (prefix,next_index) in update_combos:
            if repr(prefix) not in past_prefixes:
                past_prefixes[repr(prefix)] = {} 
            if next_index not in past_prefixes[repr(prefix)]:
                past_prefixes[repr(prefix)][next_index] = 0
            past_prefixes[repr(prefix)][next_index] = max(past_prefixes[repr(prefix)][next_index],total_value)
        
    # Step 3: Find the best action/value, and backprop 
    three_best_combos = get_best_combo(past_prefixes,budget)
    three_values = []
    combo_to_volunteers = []

    for combo in three_best_combos:
        group_indices = [0 for i in range(num_groups)]
        combo_to_volunteers.append([])
        for i in combo:
            combo_to_volunteers[-1].append(best_group_arms[i][group_indices[i]])
            group_indices[i] += 1

        value = get_total_value(env,all_match_probs,best_group_arms,state,group_indices,value_network,num_future_samples=100)
        three_values.append(value)
    
    best_value = max(three_values)
    combo_to_volunteers = combo_to_volunteers[np.argmax(three_values)]

    action = np.zeros(len(state), dtype=np.int8)
    action[combo_to_volunteers] = 1

    past_states.append(list(state))
    past_values.append(best_value)
    past_actions.append(action)

    update_value_policy_network(past_states,past_values,past_actions,value_network,criterion_value,optimizer_value,policy_network,criterion_policy,optimizer_policy,value_losses)
    memory = past_states, past_values, past_actions, value_losses, value_network, criterion_value, optimizer_value, policy_network, criterion_policy, optimizer_policy, memoizer

    return action, memory 

