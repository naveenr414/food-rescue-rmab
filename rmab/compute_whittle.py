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
from scipy.stats import binom


import heapq  # priority queue

whittle_threshold = 1e-4
value_iteration_threshold = 1e-4

def get_q_vals(transitions, state, predicted_subsidy, discount, threshold=value_iteration_threshold,reward_function='activity',lamb=0,
                        match_prob=0.5):
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
        return s*a*match_prob - a*predicted_subsidy 

    def combined_reward(s,a):
        return s*a*match_prob + lamb*s - a*predicted_subsidy 

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
    return Q_func[state,:]

def arm_value_iteration(transitions, state, predicted_subsidy, discount, threshold=value_iteration_threshold,reward_function='activity',lamb=0,
                        match_prob=0.5):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)
    """
    return np.argmax(get_q_vals(transitions,state,predicted_subsidy,discount,threshold,reward_function=reward_function,lamb=lamb,match_prob=match_prob))

def arm_value_v_iteration(transitions, state, predicted_subsidy, discount, threshold=value_iteration_threshold,reward_function='activity',lamb=0,
                        match_prob=0.5):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)
    """

    Q_vals = get_q_vals(transitions,state,predicted_subsidy,discount,threshold,reward_function=reward_function,lamb=lamb,match_prob=match_prob)
    return np.argmax(Q_vals), np.max(Q_vals)



def arm_value_iteration_exponential(all_transitions, discount, budget, volunteers_per_arm, threshold=value_iteration_threshold,reward_function='matching',lamb=0,match_probability_list=[]):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)

    Arguments:
        all_transitions: N x num_states x num_actions (presumably 2) x num_states
        discount: Gamma, float

    Returns: Q_func, numpy matrix with Q values for each combination of states, 
        and each combination of actions
        This is encoded as a 2^N x 2^N matrix, where a state is encoded in binary
    """
    assert discount < 1
    n_arms, _ = all_transitions.shape[0], all_transitions.shape[2]
    
    N = len(match_probability_list)
    num_real_states = 2**(N)

    value_func = np.random.rand(num_real_states)
    difference = np.ones((num_real_states))
    iters = 0
    match_probability_list = np.array(match_probability_list)
    
    all_s = np.array(list(product([0, 1], repeat=N)))
    all_s = [np.array(i) for i in all_s]
    
    all_a = list(combinations(range(N), budget))
    all_a = [np.array(list_to_binary(i,N)) for i in all_a]

    def reward_activity(s,a):
        return np.sum(s)

    def reward_matching(s,a):
        return (1-np.prod(np.power(1-match_probability_list,s*a)))
        
    def reward_combined(s,a):
        return (1-np.prod(np.power(1-match_probability_list,s*a))) + lamb*np.sum(s)

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
                transition_probability = np.prod([all_transitions[i//volunteers_per_arm][s[i]][a[i]][s_prime[i]]
                            for i in range(N)])
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
                action = np.zeros(N)
                for i in a:
                    action[i] = 1 

                for s_prime in all_s:
                    s_prime_rep = binary_to_decimal(s_prime)
                    Q_func[s_rep,a_rep] += precomputed_transition_probabilities[s_rep,a_rep,s_prime_rep] * (r(s,a)
                         + discount * value_func[s_prime_rep])
            value_func[s_rep] = np.max(Q_func[s_rep, :])
        difference = np.abs(orig_value_func - value_func)

    return Q_func 

def arm_value_iteration_sufficient(transitions, state, T_stat,predicted_subsidy, discount, n_arms,match_prob,budget,threshold=value_iteration_threshold,reward_function='activity',lamb=0,probs=[]):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)
    """
    assert discount < 1

    n_states, n_actions = transitions.shape
    p = match_prob

    value_func = np.random.rand(n_arms+1,n_states)
    difference = np.ones((n_states))
    iters = 0

    # lambda-adjusted reward function
    def reward(T,s, a):
        return s - a * predicted_subsidy

    def reward_matching(T,s,a):
        return s*a - a*predicted_subsidy 

    def combined_reward(T,s,a):
        if T>0 and a == s == 1:
            orig_T = T
            T = min(T,budget)
            return (1-np.power(1-p,T))/orig_T + lamb*s - a*predicted_subsidy 
            # return (1-np.power(1-p,T) - (1-np.power(1-p,T-1))) + lamb*s - a*predicted_subsidy
        elif T == 0 and a==s==1:
            return 0
        else:
            return s*a + lamb*s - a*predicted_subsidy 

    # binom_pmfs = [1,1,1,1,1,1,1]
    # binom_pmfs = np.array(binom_pmfs)/np.sum(binom_pmfs)
    # binom_pmfs = np.array([binom.pmf(i,n_arms,0.5) for i in range(n_arms+1)])
    # binom_pmfs /= np.sum(binom_pmfs)

    # for i in range(len(a_0)):
    #     for j in range(len(a_1)):
    #         binom_pmfs[i+j] += a_0[i]*a_1[j]
    
    # binom_pmfs = np.array(binom_pmfs)/np.sum(binom_pmfs)
    
    if probs != []:
        binom_pmfs = probs 
    else:
        binom_pmfs = [1/(n_arms+1) for i in range(n_arms+1)]


    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        # calculate Q-function
        Q_func = np.zeros((n_arms+1,n_states, n_actions))

        weights = np.random.rand(n_arms+1)
        weights /= weights.sum()


        for T in range(n_arms+1):
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
                        

                    for s_prime in range(n_states):
                        if s_prime == 0:
                            p_s = 1-transitions[s,a]
                        else:
                            p_s = transitions[s,a]

                        for T_prime in range(n_arms+1):
                            # p_T = T**(T_prime)*np.exp(-T)/(np.math.factorial(T_prime)) # Poisson(T)  
                            # if s_prime > T_prime:
                            #     p_T = 0
                            # else:                           
                            #     p_T = (T-s)**(T_prime-s_prime)*np.exp(-(T-s))/(np.math.factorial(T_prime-s_prime)) # Poisson(T)                             
                            # p_T = weights[T_prime]
                            # p_T = 1/(n_arms+1)
                            #p_T = 2**(T_prime)*np.exp(-2)/(np.math.factorial(T_prime))
                            p_T = binom_pmfs[T][T_prime]                            
                            Q_func[T,s, a] += p_s*p_T * (r(T,s, a) + discount * value_func[T_prime,s_prime])

                value_func[T,s] = np.max(Q_func[T,s, :])

        difference = np.abs(orig_value_func - value_func)

    # print(f'q values {Q_func[state, :]}, action {np.argmax(Q_func[state, :])}')
    return np.argmax(Q_func[T_stat,state, :])

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

def get_init_bounds(transitions,lamb=0):
    lamb = max(lamb,1)+1
    lb = -lamb
    ub = lamb
    return lb, ub

def arm_compute_whittle_sufficient(transitions, state, T_stat,discount, subsidy_break, n_arms,match_prob,budget,eps=whittle_threshold,reward_function='activity',lamb=0,probs=[]):
    """
    compute whittle index for a single arm using binary search

    subsidy_break = the min value at which we stop iterating

    param transitions:
    param eps: epsilon convergence
    returns Whittle index
    """
    lb, ub = get_init_bounds(transitions,lamb) # return lower and upper bounds on WI
    top_WI = []
    while abs(ub - lb) > eps:
        predicted_subsidy = (lb + ub) / 2
        # print('lamb', lamb_val, lb, ub)

        # we've already filled our knapsack with higher-valued WIs
        if ub < subsidy_break:
            # print('breaking early!', subsidy_break, lb, ub)
            return -10

        action = arm_value_iteration_sufficient(transitions, state, T_stat,predicted_subsidy, discount,n_arms,match_prob,budget,reward_function=reward_function,lamb=lamb,probs=probs)
        if action == 0:
            # optimal action is passive: subsidy is too high
            ub = predicted_subsidy
        elif action == 1:
            # optimal action is active: subsidy is too low
            lb = predicted_subsidy
        else:
            raise Exception(f'action not binary: {action}')
    subsidy = (ub + lb) / 2
    return subsidy

def arm_compute_whittle(transitions, state, discount, subsidy_break, eps=whittle_threshold,reward_function='activity',lamb=0,match_prob=0.5,match_probability_list=[],get_v=False):
    """
    compute whittle index for a single arm using binary search

    subsidy_break = the min value at which we stop iterating

    param transitions:
    param eps: epsilon convergence
    returns Whittle index
    """
    lb, ub = get_init_bounds(transitions,lamb) # return lower and upper bounds on WI
    top_WI = []

    while abs(ub - lb) > eps:
        predicted_subsidy = (lb + ub) / 2
        # print('lamb', lamb_val, lb, ub)

        # we've already filled our knapsack with higher-valued WIs
        if ub < subsidy_break:
            # print('breaking early!', subsidy_break, lb, ub)
            return -10

        if get_v:
            action, v_val = arm_value_v_iteration(transitions, state, predicted_subsidy, discount,reward_function=reward_function,lamb=lamb,
                        match_prob=match_prob)
        else:
            action = arm_value_iteration(transitions, state, predicted_subsidy, discount,reward_function=reward_function,lamb=lamb,
                        match_prob=match_prob)

        if action == 0:
            # optimal action is passive: subsidy is too high
            ub = predicted_subsidy
        elif action == 1:
            # optimal action is active: subsidy is too low
            lb = predicted_subsidy
        else:
            raise Exception(f'action not binary: {action}')
    
    threshold = value_iteration_threshold
    # print(predicted_subsidy,np.mean(transitions),state,match_prob,get_q_vals(transitions,state,0,discount,threshold,reward_function=reward_function,lamb=lamb,match_prob=match_prob))

    subsidy = (ub + lb) / 2

    if get_v:
        return subsidy, v_val

    return subsidy
