import numpy as np
import random 
import matplotlib.pyplot as plt
import json 
import argparse 
import sys
from copy import deepcopy
import secrets
from itertools import combinations
from rmab.simulator import create_transitions_from_prob, run_multi_seed
from rmab.whittle_policies import whittle_index, shapley_index_custom
from rmab.compute_whittle import arm_compute_whittle_multi_state, arm_compute_whittle, fast_arm_compute_whittle, arm_value_iteration_exponential_global_transition
from collections import Counter
import gurobipy as gp
from gurobipy import GRB
from rmab.utils import Memoizer, restrict_resources

def get_q_vals_global(transitions, state, budget,predicted_subsidy, discount, threshold=1e-5,reward_function='activity',lamb=1,
                        match_prob=0,match_transitions=[],all_probs=[],num_arms=1,approach='pessimistic',shapley_value=0):
    """Get the Q values for one arm, when there is a global transition

    Arguments: 
        transitions: 2x2 numpy array; P_{i}(s,a,1)
        state: integer, which state s_{i} (0 or 1) is an arm in
        predicted_subsidy: float, w, how much to penalize pulling an arm by
        discount: float, \gamma, discount for reward
        lamb: float, \alpha, tradeoff between R_{i} and R_{glob}
        match_prob: p_{i}(s_{i}), current marginal reward
        num_arms: N, total number of arms
        shapley: Optional, shapley value for pulling an arm; 
            alternative way of estimating reward

    Returns: List of Q values for current state; [Q(s,0),Q(s,1)] 
    """
    assert discount < 1

    n_states, n_actions = transitions.shape
    value_func = np.array([random.random() for i in range(n_states)])
    difference = np.ones((n_states))
    iters = 0


    # lambda-adjusted reward function
    def reward(s, a):
        if shapley_value > 0:
            return (1-lamb)*shapley_value*s*a + lamb*s/num_arms - a*predicted_subsidy
        else:
            return lamb*s/num_arms - a*predicted_subsidy

    top_K_probs = all_probs[np.argsort(all_probs)[-budget:]]

    while np.max(difference) >= threshold:
        iters += 1
        orig_value_func = np.copy(value_func)

        # calculate Q-function
        Q_func = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                r = reward  

                # transitioning to state = 0
                if s == a == 1:
                    Q_func[s, a] += (1 - transitions[s, a]-match_prob) * (r(s, a) + discount * value_func[0])
                else:
                    Q_func[s, a] += (1 - transitions[s, a]) * (r(s, a) + discount * value_func[0])

                # # transitioning to state = 1
                Q_func[s, a] += transitions[s, a] * (r(s, a) + discount * value_func[1])

                if s == 1 and a == 1:
                    if approach == 'average':
                        prob_2 = 0
                        for i in range(budget):
                            prob_2 += (1-np.mean(all_probs))**i 
                        prob_2 *= match_prob/budget 
                    elif approach == 'optimistic':
                        prob_2 = match_prob 
                    elif approach == 'pessimistic':
                        prob_2 = 0
                        for i in range(budget):
                            prob_2 += (1-np.mean(top_K_probs))**i 
                        prob_2 *= match_prob/budget 
                    else:
                        raise Exception("{} not found".format(approach))

                    if shapley_value > 0:
                        Q_func[s,a] += prob_2 * (match_transitions[0]*discount*value_func[0]
                                                        +match_transitions[1]*discount*value_func[1])
                    else:
                        Q_func[s,a] += prob_2 * ((1-lamb) + match_transitions[0]*discount*value_func[0]
                                                        +match_transitions[1]*discount*value_func[1])
                    Q_func[s,a] += (match_prob-prob_2) * (transitions[s,a]/(1-match_prob)*discount*value_func[1]
                                                    +(1-transitions[s,a])/(1-match_prob)*discount*value_func[0])
            value_func[s] = np.max(Q_func[s, :])

        difference = np.abs(orig_value_func - value_func)

    # print(f'q values {Q_func[state, :]}, action {np.argmax(Q_func[state, :])}')
    return Q_func[state,:]


def get_whittle_index_global(arm_transitions, all_probs, state, budget, discount, min_chosen_subsidy,lamb,num_arms=1,approach="optimistic",shapley_value=0):
    whittle_low = 0
    whittle_high = 10

    while whittle_high-whittle_low > 1e-3:
        Q = get_q_vals_global(arm_transitions[:2,:,1],state,budget,(whittle_low+whittle_high)/2,discount,match_prob=arm_transitions[1,1,2],match_transitions=arm_transitions[2,1,:2],all_probs=all_probs,num_arms=num_arms,approach=approach,lamb=lamb,shapley_value=shapley_value)

        if Q[0] < Q[1]:
            whittle_low = (whittle_low+whittle_high)/2
        else:
            whittle_high = (whittle_low+whittle_high)/2

    return whittle_low 

def whittle_index_global(env,state,budget,lamb,memoizer,approach,shapley_values=None):
    """Compute the Whittle index for arms when there are global transitions
    
    Arguments:  
        env: RMAB Simulator
        state: 0-1 list for each arm
        budget: K, maximum arms to pull
        lamb: [0,1] float, balance between matching, engagement
        approach: String, either "optimistic", "average" or "pessimistic"; 
            Impacts computation of Q Values/estimating transition probabilities
            See get_q_vals_global
        shapely_values: Optional, List of Shapley Values per arm
            If None, then we use reward as being in state 2"""
    
    N = len(state)
    
    if shapley_values is None:
        shapley_values = [0 for i in range(N)]

    true_transitions = env.transitions 
    discount = env.discount 

    state_WI = np.zeros((N))
    min_chosen_subsidy = -1 
    for i in range(N):
        arm_transitions = true_transitions[i//env.volunteers_per_arm]
        check_set_val = memoizer.check_set(arm_transitions, state[i])
        if check_set_val != -1:
            state_WI[i] = check_set_val
        else:
            state_WI[i] = get_whittle_index_global(arm_transitions, true_transitions[:,1,1,2],state[i], budget,discount=discount, min_chosen_subsidy=min_chosen_subsidy,lamb=lamb,num_arms=len(state),approach=approach,shapley_value=shapley_values[i])
            memoizer.add_set(arm_transitions, state[i], state_WI[i])
    return state_WI

def global_transition_policy(env,state,budget,lamb,memory,per_epoch_results,approach):
    """Select which arms to pull based on their Whittle Index through the Global Transition
        That is, we simplify the problem to account for global transitions
        Then pull arms through Whittle policies 
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Information on previously computed Whittle indices, the memoizer
        per_epoch_results: Any information computed per epoch; unused here
        approach: String, either "optimistic", "average" or "pessimistic"; 
            Impacts computation of Q Values/estimating transition probabilities
            See get_q_vals_global

    Returns: Actions, numpy array of 0-1 for each agent, and memory is the Memoizer"""

    
    N = len(state) 

    if memory == None:
        memoizer = Memoizer('optimal')
        for i in range(2):
            s = [i for _ in range(len(state))]
            whittle_index_global(env,s,budget,lamb,memoizer,approach=approach)
    else:
        memoizer = memory 
    N = len(state) 

    state_WI = whittle_index_global(env,state,budget,lamb,memoizer,approach)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, memoizer 

def optimistic_policy(env,state,budget,lamb,memory,per_epoch_results):
    return global_transition_policy(env,state,budget,lamb,memory,per_epoch_results,approach='optimistic')
def average_policy(env,state,budget,lamb,memory,per_epoch_results):
    return global_transition_policy(env,state,budget,lamb,memory,per_epoch_results,approach='average')
def pessimistic_policy(env,state,budget,lamb,memory,per_epoch_results):
    return global_transition_policy(env,state,budget,lamb,memory,per_epoch_results,approach='pessimistic')

def shapley_whittle_global_transition_custom_policy(env,state,budget,lamb,memory, per_epoch_results):
    """Let's the reward for the global transition policy depend on Shapley values
        Rather than being in state 2
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and the Whittle memoizer"""

    N = len(state) 

    if memory == None:
        memory_whittle = Memoizer('optimal')
        memory_shapley = np.array(shapley_index_custom(env,np.ones(len(state)),{})[0])

        for i in range(2):
            s = [i for _ in range(len(state))]
            whittle_index_global(env,s,budget,lamb,memory_whittle,approach='optimistic',shapley_values=memory_shapley)
    else:
        memory_whittle, memory_shapley = memory 

    state_WI = whittle_index_global(env,state,budget,lamb,memory_whittle,'optimistic',shapley_values=memory_shapley)

    sorted_WI = np.argsort(state_WI)[::-1]
    action = np.zeros(N, dtype=np.int8)
    action[sorted_WI[:budget]] = 1

    return action, (memory_whittle, memory_shapley) 

def global_transition_linear_policy(env,state,budget,lamb,memory,per_epoch_results):
    """Compute the local rewards, and greedily optimize which arms to pull
    The variables are the arms to pull, the objective is the expected Whittle index
    By taking an average over transitioning/not transitioning to state 2
    And scaling the reward when transitioning
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and the Whittle memoizer"""

    
    N = len(state) 
    transitions = env.transitions 

    if memory == None:
        all_WI = []
        state_WI_if_2 = []

        for s in range(2):
            all_WI.append([])
            for i in range(len(state)):
                arm_transitions = deepcopy(transitions[i,:2,:,:2])
                for j in range(2):
                    arm_transitions[j,1] /= np.sum(arm_transitions[j,1])
                all_WI[-1].append(fast_arm_compute_whittle(arm_transitions[:,:,1], s, env.discount, 1e-3,reward_function='activity',lamb=1,match_prob=0,num_arms=N))
        all_WI = np.array(all_WI)

        for i in range(N):
            arm_transitions = np.zeros((2,2))
            arm_transitions[0,0] = transitions[i,0,0,1]
            arm_transitions[1,0] = transitions[i,2,0,1]
            arm_transitions[0,1] = transitions[i,0,1,1]
            arm_transitions[1,1] = transitions[i,2,1,1]
            state_WI_if_2.append(fast_arm_compute_whittle(arm_transitions, 1, env.discount, 1e-3,reward_function='activity',lamb=1,match_prob=0,num_arms=N)+1)

        memory = all_WI, state_WI_if_2

    all_WI, state_WI_if_2 = memory 
    state_WI = all_WI[state, np.arange(len(state))]

    a = [state[i]*(state_WI_if_2[i]*transitions[i,state[i],1,2] - state_WI[i]*transitions[i,state[i],1,2]) for i in range(N)]
    c = [state_WI[i] for i in range(N)]
    m = gp.Model()
    m.setParam('OutputFlag', 0)

    # Create variables
    z = m.addVars(N, lb=0, name="z")

    # Constraint: sum(b[i] * z[i]) = t
    # m.addConstr(gp.quicksum(b[i] * z[i] for i in range(N)) == t, "constraint_1")
    m.addConstr(gp.quicksum(z[i] for i in range(N)) == budget, "constraint_1")
    for i in range(N):
        m.addConstr(z[i] <= 1, name=f"c3_{i}")

    # Set objective
    m.setObjective(gp.quicksum((a[i] + c[i]) * z[i] for i in range(N)), GRB.MAXIMIZE)

    # Optimize the model
    m.optimize()
    y_opt = m.getAttr('x', z)
    action = [y_opt[i] for i in range(N)]
    action = [round(i) for i in action]
    action = np.array(action)

    return action, (all_WI,state_WI_if_2)


def q_iteration_epoch_global_transition(env,lamb,reward_function='combined'):
    """Compute Q Values for all combinations of agents in a given environment
    
    Arguments:
        env: RMAB Simulator environment
        lamb: \alpha, tradeoff between R_{i} and R_{glob}
        
    Returns: Q values, one for each combination of state + action"""

    match_probability = env.match_probability_list 
    if match_probability != []:
        match_probability = np.array(match_probability)[env.agent_idx]
    true_transitions = env.transitions
    discount = env.discount 
    budget = env.budget 

    Q_vals = arm_value_iteration_exponential_global_transition(true_transitions,discount,budget,env.volunteers_per_arm,env.reward_type,env.reward_parameters,
                    reward_function=reward_function,lamb=lamb,
                    match_probability_list=match_probability)

    return Q_vals 

def q_iteration_custom_epoch_global_transition():
    """Run Q Iteration with a custom reward function: 
    
    Arguments: None
    
    Returns: A policy which runs q_iteration using the custom reward function"""
    def q_iteration(env,lamb):
        return q_iteration_epoch_global_transition(env,lamb,reward_function='activity')
    return q_iteration
