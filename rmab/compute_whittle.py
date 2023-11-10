"""
standard Whittle index computation based on binary search

POSSIBLE OPTIMIZATIONS TO HELP SPEED
- keep track of top k WIs so far. then in future during binary search, if we go below that WI just quit immediately
"""

import numpy as np
from itertools import product, combinations
from rmab.utils import binary_to_decimal, list_to_binary
import torch
import torch.nn as nn
import torch.optim as optim


import heapq  # priority queue

whittle_threshold = 1e-4
value_iteration_threshold = 1e-2

def arm_value_iteration(transitions, state, predicted_subsidy, discount, threshold=value_iteration_threshold,reward_function='activity',lamb=0):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)
    """
    assert discount < 1

    n_states, n_actions = transitions.shape
    value_func = np.random.rand(n_states)
    difference = np.ones((n_states))
    iters = 0

    # lambda-adjusted reward function
    def reward(s, a):
        return s - a * predicted_subsidy

    def reward_matching(s,a):
        return s*a - a*predicted_subsidy 

    def combined_reward(s,a):
        return s*a + lamb*s - a*predicted_subsidy 

    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        # calculate Q-function
        Q_func = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                if reward_function == 'activity':
                    r = reward  
                elif reward_function == 'matching':
                    r = reward_matching 
                elif reward_function == 'combined':
                    r = combined_reward
                else:
                    raise Exception("Reward function {} not found".format(reward_function))
                     
                # transitioning to state = 0
                Q_func[s, a] += (1 - transitions[s, a]) * (r(s, a) + discount * value_func[0])

                # transitioning to state = 1
                Q_func[s, a] += transitions[s, a] * (r(s, a) + discount * value_func[1])

            value_func[s] = np.max(Q_func[s, :])

        difference = np.abs(orig_value_func - value_func)

    # print(f'q values {Q_func[state, :]}, action {np.argmax(Q_func[state, :])}')
    return np.argmax(Q_func[state, :])

def arm_value_iteration_exponential(all_transitions, discount, budget, match_prob, threshold=value_iteration_threshold,reward_function='matching',lamb=0):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)

    Arguments:
        all_transitions: N x num_states x num_actions (presumably 2) x num_states
        discount: Gamma, float
        match_prob: Float, match probability for each arm

    Returns: Q_func, numpy matrix with Q values for each combination of states, 
        and each combination of actions
        This is encoded as a 2^N x 2^N matrix, where a state is encoded in binary
    """
    assert discount < 1
    n_arms, _ = all_transitions.shape[0], all_transitions.shape[2]
    num_real_states = 2**(n_arms)
    value_func = np.random.rand(num_real_states)
    difference = np.ones((num_real_states))
    iters = 0
    p = match_prob 
    
    all_s = np.array(list(product([0, 1], repeat=n_arms)))
    all_s = [np.array(i) for i in all_s]
    
    all_a = list(combinations(range(n_arms), budget))
    all_a = [np.array(list_to_binary(i,n_arms)) for i in all_a]

    def reward_activity(s,a):
        return np.sum(s)

    def reward_matching(s,a):
        return (1-np.power(1-p,s.dot(a)))
        
    def reward_combined(s,a):
        return (1-np.power(1-p,s.dot(a))) + lamb*np.sum(s)

    if reward_function == 'activity':
        r = reward_activity
    elif reward_function == 'matching':
        r = reward_matching 
    elif reward_function == 'combined': 
        r = reward_combined 
    else:
        raise Exception("{} reward function not found".format(reward_function))

    # Precompute transition probabilities for speed 
    precomputed_transition_probabilities = np.zeros((num_real_states,num_real_states,num_real_states))
    for s in all_s:
        s_rep = binary_to_decimal(s) 

        for s_prime in all_s:
            s_prime_rep = binary_to_decimal(s_prime) 

            for a in all_a:
                a_rep = binary_to_decimal(a)
                transition_probability = np.prod([all_transitions[i][s[i]][a[i]][s_prime[i]]
                            for i in range(n_arms)])
                precomputed_transition_probabilities[s_rep][a_rep][s_prime_rep] = transition_probability
    

    # Perform Q Iteration 
    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        Q_func = np.zeros((num_real_states, num_real_states))
        for s in all_s:
            s_rep = binary_to_decimal(s) 

            for a in all_a:
                a_rep = binary_to_decimal(a)
                action = np.zeros(n_arms)
                for i in a:
                    action[i] = 1 

                for s_prime in all_s:
                    s_prime_rep = binary_to_decimal(s_prime)
                    Q_func[s_rep,a_rep] += precomputed_transition_probabilities[s_rep,a_rep,s_prime_rep] * (r(s,a)
                         + discount * value_func[s_prime_rep])
            value_func[s_rep] = np.max(Q_func[s_rep, :])
        difference = np.abs(orig_value_func - value_func)

    return Q_func 

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def arm_value_iteration_neural(all_transitions, discount, budget, match_prob, threshold=value_iteration_threshold,reward_function='matching',lamb=0):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)

    Arguments:
        all_transitions: N x num_states x num_actions (presumably 2) x num_states
        discount: Gamma, float
        match_prob: Float, match probability for each arm

    Returns: Q_func, numpy matrix with Q values for each combination of states, 
        and each combination of actions
        This is encoded as a 2^N x 2^N matrix, where a state is encoded in binary
    """
    assert discount < 1
    n_arms, _ = all_transitions.shape[0], all_transitions.shape[2]
    num_real_states = 2**(n_arms)
    p = match_prob 
    
    def reward_activity(s,a):
        return torch.pow(s)

    def reward_matching(s,a):
        return (1-torch.pow(1-p,s.dot(a)))
        
    def reward_combined(s,a):
        if torch.sum(a) > budget:
            return -10000

        return (1-torch.pow(1-p,s.dot(a))) + lamb*torch.sum(s)

    if reward_function == 'activity':
        r = reward_activity
    elif reward_function == 'matching':
        r = reward_matching 
    elif reward_function == 'combined': 
        r = reward_combined 
    else:
        raise Exception("{} reward function not found".format(reward_function))

    input_size = n_arms
    output_size = 2**n_arms
    q_network = QNetwork(input_size, output_size)


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    q_network = q_network.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)

    num_episodes = 500

    for episode in range(num_episodes):
        state = torch.randint(0, 2, (n_arms,), dtype=torch.float32) 
        total_reward = 0
        done = False

        trials = 50 
        for trial in range(trials):
            state = state.to(device)
            # Choose an action using epsilon-greedy policy
            if np.random.rand() < 0.05:
                action = [1 for i in range(budget)] + [0 for i in range(n_arms-budget)]
                binary_action = binary_to_decimal(action)
                np.random.shuffle(action)
                action = torch.Tensor(action).to(device)
            else:
                q_values = q_network(state)
                binary_action = torch.argmax(q_values).item()
                binary_val = bin(binary_action)[2:].zfill(n_arms)
                action = np.zeros(n_arms, dtype=np.int8)
                action = torch.Tensor(np.array([int(i) for i in binary_val])).to(device)


            # Simulate the environment
            next_state = []

            for i in range(n_arms):
                current_state = state[i] 
                one_probability = all_transitions[i][int(current_state.item())][int(action[i].item())][1]
                next_state.append(int(np.random.random()<one_probability))

            next_state = torch.Tensor(np.array(next_state))
            reward = r(state,action)

            # Compute the target Q-value using the Q-learning formula
            with torch.no_grad():
                target_q_values = q_network(next_state.to(device))
                target_q_value = reward + discount * torch.max(target_q_values)
                target_q_value = target_q_value.to(device)

            # Compute the loss and perform a gradient descent step
            optimizer.zero_grad()
            current_q_values = q_network(state)
            loss = criterion(current_q_values[binary_action], target_q_value)
            loss.backward()
            optimizer.step()

            state = next_state

    return q_network 



def arm_value_iteration_approximate(all_transitions, discount, budget, match_prob, threshold=value_iteration_threshold,reward_function='matching',lamb=0,arm_num=0):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)

    Arguments:
        all_transitions: N x num_states x num_actions (presumably 2) x num_states
        discount: Gamma, float
        match_prob: Float, match probability for each arm

    Returns: Q_func, numpy matrix with Q values for each combination of states, 
        and each combination of actions
        This is encoded as a 2^N x 2^N matrix, where a state is encoded in binary
    """
    assert discount < 1
    n_arms,n_states, n_actions,_ = all_transitions.shape
    num_actions = 2**(n_arms)
    value_func = np.random.rand(n_arms,n_arms+1)
    difference = np.ones((n_arms,num_actions,budget+1,n_arms+1))
    iters = 0
    p = match_prob 
        
    all_a = list(combinations(range(n_arms), budget))
    all_a = [np.array(list_to_binary(i,n_arms)) for i in all_a]

    def reward_difference(s,a,A,S,T):
        return (np.power(1-p,S-s*a)-np.power(1-p,S) + s)    

    # Perform Q Iteration 
    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        Q_func = np.zeros((n_states, 2,num_actions,budget+1,n_arms+1))
        for s in range(n_states):
            for q,a in enumerate(all_a):
                a_rep = binary_to_decimal(a)
                action = np.array(a)

                num_by_S_T = {}
                tot_by_S_T = {}
                num_pathways = 50
                for i in range(num_pathways):
                    other_states = [np.random.randint(0,2) for i in range(n_arms)]
                    other_states[arm_num] = s

                    S = np.dot(other_states,action)
                    T = int(np.round(sum([all_transitions[k][other_states[k]][action[k]][1] for k in range(n_arms)])))
                    reward = reward_difference(s,action[arm_num],action,S,T)


                    if (S,T) not in tot_by_S_T:
                        tot_by_S_T[(S,T)] = 0
                        num_by_S_T[(S,T)] = 1
                    
                    num_by_S_T[(S,T)] += 1
                    tot_by_S_T[(S,T)] += reward 


                    for s_prime in range(n_states):
                        for T_prime in range(n_arms+1):
                            s_prime_prob = all_transitions[arm_num][s][action[arm_num]][s_prime]
                            T_prime_prob = T**(T_prime)*np.exp(-T)/(np.math.factorial(T_prime))


                            tot_by_S_T[(S,T)] += discount*s_prime_prob*T_prime_prob*value_func[s_prime,T_prime] 

                for (S,T) in num_by_S_T:    
                    Q_func[s,action[arm_num],a_rep,S,T] = tot_by_S_T[(S,T)]/num_by_S_T[(S,T)] 

                

            for T in range(n_arms+1):
                value_func[s,T] = np.max(Q_func[s,:,:,:,T])
        difference = np.abs(orig_value_func - value_func)

    return Q_func 


def get_init_bounds(transitions):
    lb = -1
    ub = 1
    return lb, ub

def arm_compute_whittle(transitions, state, discount, subsidy_break, eps=whittle_threshold,reward_function='activity',lamb=0):
    """
    compute whittle index for a single arm using binary search

    subsidy_break = the min value at which we stop iterating

    param transitions:
    param eps: epsilon convergence
    returns Whittle index
    """
    lb, ub = get_init_bounds(transitions) # return lower and upper bounds on WI
    top_WI = []
    while abs(ub - lb) > eps:
        predicted_subsidy = (lb + ub) / 2
        # print('lamb', lamb_val, lb, ub)

        # we've already filled our knapsack with higher-valued WIs
        if ub < subsidy_break:
            # print('breaking early!', subsidy_break, lb, ub)
            return -10

        action = arm_value_iteration(transitions, state, predicted_subsidy, discount,reward_function=reward_function,lamb=lamb)
        if action == 0:
            # optimal action is passive: subsidy is too high
            ub = predicted_subsidy
        elif action == 1:
            # optimal action is active: subsidy is too low
            lb = predicted_subsidy
        else:
            raise Error(f'action not binary: {action}')
    subsidy = (ub + lb) / 2
    return subsidy
