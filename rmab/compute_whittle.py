import numpy as np
from itertools import product, combinations
from rmab.utils import binary_to_decimal, list_to_binary, custom_reward
import random 
import markovianbandit as bandit

value_iteration_threshold = 1e-6

def fast_compute_whittle_indices(transitions,rewards,discount,computed_values={}):
    """Compute the Whittle indices using the markovianbandit library
    
    Arguments:
        Transitions: Numpy aray, transitions for one arm, of size Sx2xS
        rewards: Numpy array, rewards for one arm, of size Sx2
        discount: \gamma, discount factor; between [0,1]
        computed_values: Dictionary which stores previously computed Whittle indices
    
    Returns: Float, the Whittle index for this arm"""

    P0 = transitions[:,0,:]
    P1 = transitions[:,1,:]
    R0 = rewards[:,0]
    R1 = rewards[:,1]

    represent = "{}{}{}{}".format(hash(P0.tostring()),hash(P1.tostring()),hash(R0.tostring()),hash(R1.tostring()))
    if represent in computed_values:
        return computed_values[represent]

    model = bandit.restless_bandit_from_P0P1_R0R1(P0,P1,R0,R1)

    try:
        library_comp = model.whittle_indices(discount=discount,check_indexability=False)
    except Exception as e: 
        print("Error in Whittle {}".format(e))
        library_comp = [-1 for i in range(len(R1))]

    computed_values[represent] = library_comp 
    return library_comp 

def arm_value_iteration_exponential(all_transitions, discount, budget, volunteers_per_arm, reward_type,reward_parameters,threshold=value_iteration_threshold,reward_function='matching',lamb=0,power=None,match_probability_list=[]):
    """Run Q Iteration for the exponential state space of all state/action combinations 

    Arguments:
        all_transitions: Numpy array of size (N,|S|,|A|,|S|)
        discount: \gamma, float
        budget: K, integer, max arms to pull
        volunteers_per_arm: When using multiple arms with the same transition probability
            We have volunteers_per_arm arms with all_transitions[0], volunteers_per_arm with all_transitions[1], etc.
        reward_type: Either 'activity' (maximize for R_{i}(s_{i},a_{i})) or 'custom' (maximize for R(s,a))
        reward_parameters: Dictionary with three keys: arm_set_low, arm_set_high, and universize_size (m)
            This defines the range of m_{i} or Y_{i} values for reward functions
        lamb: float, \alpha, tradeoff between R_{i} and R_{glob}
        match_probability_list: The m_{i} or Y_{i} values 

    Returns: Q_func, numpy matrix with Q values for each combination of states, 
        and each combination of actions
        This is encoded as a 2^N x 2^N matrix, where a state is encoded in binary
    """
    assert discount < 1
    n_arms, _ = all_transitions.shape[0], all_transitions.shape[2]
    
    N = len(match_probability_list)
    num_real_states = 2**(N)
    value_func = np.array([random.random() for i in range(num_real_states)])
    difference = np.ones((num_real_states))
    iters = 0
    match_probability_list = np.array(match_probability_list)
    
    all_s = np.array(list(product([0, 1], repeat=N)))
    all_s = [np.array(i) for i in all_s]
    
    all_a = []
    for b in range(budget+1):
        all_a += list(combinations(range(N), b))    

    all_a = [np.array(list_to_binary(i,N)) for i in all_a]

    def reward_activity(s,a):
        return np.sum(s)

    def reward_matching(s,a):
        return (1-np.prod(np.power(1-match_probability_list,s*a)))
        
    def reward_combined(s,a):
        rew = (1-np.prod(np.power(1-match_probability_list,s*a)))*(1-lamb) + lamb*np.sum(s)/len(s)
        return rew
    
    def reward_custom(s,a):
        val = custom_reward(s,a,match_probability_list,reward_type,reward_parameters,[1])*(1-lamb) + lamb*np.sum(s)/len(s)
        return val 

    if reward_function == 'activity':
        r = reward_activity
    elif reward_function == 'matching':
        r = reward_matching 
    elif reward_function == 'combined': 
        r = reward_combined 
    elif reward_function == 'custom': 
        r = reward_custom
    else:
        raise Exception("{} reward function not found".format(reward_function))

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

                for s_prime in all_s:
                    s_prime_rep = binary_to_decimal(s_prime)
                    Q_func[s_rep,a_rep] += precomputed_transition_probabilities[s_rep,a_rep,s_prime_rep] * (r(s,a)
                         + discount * value_func[s_prime_rep])
            value_func[s_rep] = np.max(Q_func[s_rep, :])
        difference = np.abs(orig_value_func - value_func)
    return Q_func 

def Q_multi_prob(transitions, state, predicted_subsidy, discount, threshold=value_iteration_threshold,reward_function='combined',lamb=0.5,
                        match_prob=0.5,match_prob_now=1,num_arms=1,active_states=[1],p_matrix=None):
    """Value iteration when initial and subsequent rewards differ
        We do this through Q Iteration on 3 states
        State 0 and 1 are as normal, while 
        State 2 assumes we get match_prob_now reward, then we transition
        to state 0 or 1, with the same transitions as state 1

    Arguments:
        transitions: Numpy array of size 2x2 (for 2 states x 2 actions)
        state: Integer; which state the arm is currently in
            Will compute Q(s,0) and Q(s,1)
        predicted_subsidy: Float, how much to penalize pulling an arm by
        discount: Float, \gamma, how much to discount future rewards
        threshold: Loop exit condition; when value error <= threshold
            we break
        match_prob: Float; how much reward we get when pulling an arm
            in later time steps
        match_prob_now: Float; how much reward we get when pulling an arm
            in this time step
        num_arms: Total number of arms, N
    
    Returns: 3x2 Numpy array, where Q(2,a) is 
        the rewards for pulling/not pulling an arm now
    """
    assert discount < 1
    n_states, n_actions, _ = transitions.shape

    new_transitions = np.zeros((n_states+1,2,n_states))
    new_transitions[:n_states,:,:n_states] = transitions 
    new_transitions[n_states,:] = transitions[state]
    transitions = new_transitions 

    value_func = np.array([random.random() for i in range(n_states)])
    difference = np.ones((n_states))
    iters = 0

    def combined_reward(s,a):
        if s == n_states:
            s = state 
            rew = a*match_prob_now*(1-lamb) + lamb*int(s in active_states)/num_arms - a*predicted_subsidy 
        else:
            rew = p_matrix[s]*a*(1-lamb) + lamb*int(s in active_states)/num_arms - a*predicted_subsidy 
        return rew 

    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        # calculate Q-function
        Q_func = np.zeros((n_states+1, n_actions))
        for s in range(n_states+1):
            for a in range(n_actions):
                r = combined_reward   

                for s_prime in range(n_states):
                    Q_func[s,a] += transitions[s,a,s_prime] * (r(s,a) + discount * value_func[s_prime])                 

            if s < n_states:
                value_func[s] = np.max(Q_func[s, :])

        difference = np.abs(orig_value_func - value_func)

    # print(f'q values {Q_func[state, :]}, action {np.argmax(Q_func[state, :])}')
    return Q_func
