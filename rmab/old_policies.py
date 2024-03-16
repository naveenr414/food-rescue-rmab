import numpy as np
import heapq

from rmab.uc_whittle import Memoizer
from rmab.compute_whittle import arm_compute_whittle, arm_value_iteration_exponential, arm_value_iteration_neural, arm_compute_whittle_sufficient
from rmab.utils import get_stationary_distribution, binary_to_decimal, list_to_binary
from rmab.mcts_policies import run_mcts, two_step_idx_to_action
from itertools import combinations

import torch
import torch.nn as nn
import torch.optim as optim
import itertools 
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from scipy.stats import binom
from copy import deepcopy
from mcts import mcts
import random 


class VolunteerStateTwoStep():
    def __init__(self):
        self.non_match_prob = 1
        self.previous_pulls = []
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
        for i in range(len(self.states)):
            if possibleActions.count(i) < self.volunteers_per_arm:
                possibleActions.append(Action(arm=i))
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.previous_pulls.append(action.arm)
        newState.memoizer = self.memoizer
        
        return newState

    def isTerminal(self):
        if len(self.previous_pulls) == self.budget:
            return True 
        return False 

    def get_current_reward(self,state,previous_pulls):
        policy = 'index'

        expected_active_rate = 0
        non_match_prob = 1 
        for i in range(len(state)):
            action_count = previous_pulls.count(i)
            num_one = action_count 
            num_zero = self.volunteers_per_arm - num_one 
            expected_active_rate += self.transitions[i,state[i],1,1]*num_one
            expected_active_rate += self.transitions[i,state[i],0,1]*num_zero 

            non_match_prob *= (1-state[i]*self.arm_values[i])**(num_one)

        if policy == 'reward':
            value = 1-non_match_prob  + self.lamb*expected_active_rate
        elif policy == 'index':
            state_WI = np.zeros(len(state))

            for i in range(len(state_WI)):
                arm_transitions = self.env.transitions[i//self.volunteers_per_arm, :, :, 1]
                check_set_val = self.memoizer.check_set(arm_transitions, state[i])

                if check_set_val != -1:
                    state_WI[i] = check_set_val
                    
                else:
                    state_WI[i] = get_q_vals(arm_transitions, state[i], 0, self.discount,reward_function="activity",lamb=self.lamb)[1]
                    self.memoizer.add_set(arm_transitions, state[i], state_WI[i])

            # state_WI = whittle_index(env,state,budget,lamb,memoizer,reward_function="combined")
            non_match_prob = 1 
            for i in previous_pulls:
                non_match_prob *= (1-self.env.transitions[i//self.volunteers_per_arm,state[i],1,1]*self.arm_values[i])

            value = (1-non_match_prob)/(1-self.discount) + self.lamb*np.sum(state_WI[list(previous_pulls)]) 
                
        return value 
    
    def getReward(self):
        total_value = 0
        disc = 1 

        previous_rewards = []
        previous_rewards.append(self.get_current_reward(self.states,self.previous_pulls))

        action = [self.previous_pulls.count(i) for i in range(len(self.states))]

        for _ in range(self.max_depth-1):
            next_states = np.zeros(self.cohort_size,dtype=int)
            for i in range(self.cohort_size):
                num_one = action[i] 
                num_zero = self.volunteers_per_arm - num_one 
                prob = self.transitions[i, self.states[i], 0, :] * num_zero 
                prob += self.transitions[i, self.states[i], 1, :] * num_one 
                prob /= self.volunteers_per_arm

                prob = np.clip(prob,0,1)
                prob /= np.sum(prob)

                next_state = np.random.choice(a=2, p=prob)
                next_states[i] = next_state
            
            original_list = [i for i in range(self.cohort_size) for _ in range(self.volunteers_per_arm)]
            random.shuffle(original_list)

            reward = self.get_current_reward(next_states, original_list[:self.budget])
            previous_rewards.append(reward)

        for i in previous_rewards:
            total_value += disc*i 
            disc *= self.discount 

        return total_value 

    def set_state(self,states):
        self.states = np.array([int(np.round(np.mean(states[i*self.volunteers_per_arm:(i+1)*self.volunteers_per_arm]))) for i in range(self.cohort_size)])

def findMaxChild(root,explorationValue):
    bestVal = 0

    if root.state.isTerminal():
        return root.totalReward / root.numVisits + explorationValue * math.sqrt(
        2 * math.log(root.parent.numVisits) / root.numVisits)

    for child in root.children.values():
        bestVal = max(bestVal,findMaxChild(child,explorationValue))

    return bestVal 

def two_step_idx_to_action(selected_idx,env,state,lamb,memory):
    """Run MCTS to find the groups, then run MCTS again to find
        which volunteers in each group to notify
        
    Arguments:
        env: RMAB Simulator Environment
        state: 0-1 numpy array for the state
        lamb: Float, what \lambda value
        memory: List with the first element as a dictionary
            second element as a memoizer
            
    Returns: 0-1 numpy array with the actions to take"""


    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    match_probabilities *= (np.array(state)+.01*(1-np.array(state)))
    num_by_cohort = [0 for i in range(env.cohort_size)]

    for i in selected_idx:
        num_by_cohort[i] += 1
        
    selected_agents = []

    for i in range(len(num_by_cohort)):
        if num_by_cohort[i] > 0:
            k_largest = np.argsort(match_probabilities[i*env.volunteers_per_arm:(i+1)*env.volunteers_per_arm])[-num_by_cohort[i]:]
            k_largest = [i*env.volunteers_per_arm + j for j in k_largest]
            selected_agents += k_largest 
    
    action = np.zeros(len(state), dtype=np.int8)
    action[selected_agents] = 1

    return action, memory


class mcts_max():
    def __init__(self, timeLimit=None, iterationLimit=None, explorationConstant=1 / math.sqrt(2),
                 rolloutPolicy=randomPolicy):
        if timeLimit != None:
            if iterationLimit != None:
                raise ValueError("Cannot have both a time limit and an iteration limit")
            # time taken for each MCTS search in milliseconds
            self.timeLimit = timeLimit
            self.limitType = 'time'
        else:
            if iterationLimit == None:
                raise ValueError("Must have either a time limit or an iteration limit")
            # number of iterations of the search
            if iterationLimit < 1:
                raise ValueError("Iteration limit must be greater than one")
            self.searchLimit = iterationLimit
            self.limitType = 'iterations'
        self.explorationConstant = explorationConstant
        self.rollout = rolloutPolicy

    def search(self, initialState):
        self.root = treeNode(initialState, None)

        if self.limitType == 'time':
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for i in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0)
        return self.getAction(self.root, bestChild)

    def executeRound(self):
        node = self.selectNode(self.root)
        reward = self.rollout(node.state)
        self.backpropogate(node, reward)

    def selectNode(self, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant)
            else:
                return self.expand(node)
        return node

    def expand(self, node):
        actions = node.state.getPossibleActions()
        for action in actions:
            if action not in node.children.keys():
                newNode = treeNode(node.state.takeAction(action), node)
                node.children[action] = newNode
                if len(actions) == len(node.children):
                    node.isFullyExpanded = True
                return newNode

        raise Exception("Should never reach here")

    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []

        for child in node.children.values():
            # TODO: Incorporate this back in 
            nodeValue = findMaxChild(child,explorationValue)

            if nodeValue > bestValue:
                bestValue = nodeValue
                bestNodes = [child]
            elif nodeValue == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)

    def getAction(self, root, bestChild):
        for action, node in root.children.items():
            if node is bestChild:
                return action

def mcts_max_policy(env,state,budget,lamb,memory,per_epoch_results):
    return run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts_max,VolunteerState,one_step_idx_to_action)


def optimal_neural_q_iteration(env, budget, match_prob,n_episodes, n_epochs, discount,reward_function='matching',lamb=0):
    """Q-iteration to solve which arms to pull
        Doesn't decompose the arms, but can achieve higher performance
        Is quite slow 

    More details in arm_value_iteration_match function
     """
    T         = env.episode_len * n_episodes

    obs, info = env.reset(options={'reset_type': 'full'})

    all_reward = np.zeros((n_epochs, T))
    total_active = 0 

    for epoch in range(n_epochs):
        if epoch != 0: 
            env.reset(options={'reset_type': 'instance'})

        print('first state', obs)

        net = Net(env,device='cuda')
        agent = Agent(net)
        agent.train(total_time_steps=1000)
        obs, info = env.reset(options={'reset_type': 'to_0'})

        for t in range(0, T):
            # select optimal action based on known transition probabilities
            # compute whittle index for each arm
            action, _ = agent.act(obs)

            obs, reward, done, total_active_temp = env.step(action)
            
            if np.sum(action)<=budget:
                reward = 1-np.power(1-match_prob,obs.flatten().dot(action.flatten()))

            else:
                reward = 0

            # if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward
        total_active += total_active_temp[0]['final_info']['total_active']

    return all_reward, total_active/(all_reward.size*action.size)


def optimal_whittle_sufficient(env, n_episodes, n_epochs, discount,reward_function='activity',lamb=0):
    """Whittle index policy based on computing the subsidy for each arm
    This approximates the problem as the sum of Linear rewards, then 
    Decomposes the problem into the problem for each arm individually
    
    reward_function: String, either
        activity: Maximize the total activity, s_i 
        matching: Maximize the total number of matches, s_i*a_i 
        combined: Maximize s_i*a_i + \lambda s_i
    lamb: Float, hyperparameter for the combined matching """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes
    match_prob = env.match_probability

    env.reset_all()

    memoizer = Memoizer('optimal')

    all_reward = np.zeros((n_epochs, T))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        true_transitions = env.transitions
        print('first state', env.observe())
        
        p_0_0_1 = 0.3 # np.mean(true_transitions[:,0,0,1])
        p_1_1_1 = 0.7 # np.mean(true_transitions[:,1,1,1])
        p_1_0_1 = 0.6 # np.mean(true_transitions[:,1,0,1])

        binom_pmfs = [[0 for i in range(N+1)] for i in range(N+1)]

        for num_1 in range(0,N+1):
            num_0 = N-num_1 

            for i_0_0_1 in range(num_0+1):
                for i_1_1_1 in range(0,min(num_1,budget)+1):
                    for i_1_0_1 in range(0,num_1-i_1_1_1+1):
                        p_a = binom.pmf(i_0_0_1,num_0,p_0_0_1)
                        p_b = binom.pmf(i_1_1_1,min(num_1,budget),p_1_1_1)
                        p_c = binom.pmf(i_1_0_1,num_1-i_1_1_1,p_1_0_1)
                        binom_pmfs[num_1][i_0_0_1+i_1_1_1+i_1_0_1] += p_a*p_b*p_c 
        
        # for i in range(len(a_0)):
        #     for j in range(len(a_1)):
        #         for k in range(len(a_2)):

        #         binom_pmfs[i+j] += a_0[i]*a_1[j]
        
        for t in range(0, T):
            state = env.observe()
            state_str = [str(i) for i in state]
            T_stat = np.sum(state)

            # select optimal action based on known transition probabilities
            # compute whittle index for each arm
            state_WI = np.zeros(N)
            top_WI = []
            min_chosen_subsidy = -1 #0

            for i in range(N):
                arm_transitions = true_transitions[i, :, :, 1]

                # memoize to speed up
                check_set_val = memoizer.check_set(arm_transitions, str(state[i])+ ' '+str(T_stat))
                if check_set_val != -1:
                    state_WI[i] = check_set_val
                else:
                    state_WI[i] = arm_compute_whittle_sufficient(arm_transitions, state[i], T_stat,discount, min_chosen_subsidy,N,match_prob,budget,reward_function=reward_function,lamb=lamb,probs=binom_pmfs)
                    memoizer.add_set(arm_transitions, str(state[i])+' '+str(T_stat), state_WI[i])

                if len(top_WI) < budget:
                    heapq.heappush(top_WI, (state_WI[i], i))
                else:
                    # add state_WI to heap
                    heapq.heappushpop(top_WI, (state_WI[i], i))
                    min_chosen_subsidy = top_WI[0][0]  # smallest-valued item

            # pull K highest indices
            sorted_WI = np.argsort(state_WI)[::-1]

            action = np.zeros(N, dtype=np.int8)
            action[sorted_WI[:budget]] = 1

            _, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward

    return all_reward

def myopic_match_n_step(env, n_episodes, n_epochs, discount,n_step):
    """Compute the greedy policy for matching, alerting those who are in 
    the good state n_steps in the future first, then alert those n_steps+1 next, etc. 
    Arguments
        env: RMAB simulator
        n_episodes: Integer, number of episodes
        n_epochs: Integer, number of epochs
        discount: Float, gamma
        n_steps: Integer, number of steps to lookahead
            If 0, then we alert those in s=1
            If 1, we predict who will be in s=1 next
            If s=-1, then we use the steady state (s=\infinity)
            Otherwise, we use transition_matrix^n_steps
    Returns: Rewards for each epoch/episode"""
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        
        if n_step == 0 or n_step == 1:
            power_transitions = [t[:,1,:] for t in env.transitions]
        # Look ahead more than 1 step 
        elif n_step > 1:
            power_transitions = [np.linalg.matrix_power(t[:,1,:],n_step) for t in env.transitions] 
        # Look ahead infinite steps (steady state)
        elif n_step == -1:
            stationary_distribution = [get_stationary_distribution(t[:,1,:]) for t in env.transitions]
            power_transitions = [[r,r] for r in stationary_distribution]
 
        print('first state', env.observe())

        for t in range(0, T):
            state = env.observe()

            # select optimal action based on known transition probabilities
            # compute whittle index for each arm

            greedy_transitions = [i for i in range(len(state)) if state[i] == 1]
            lookahead_transitions = [(i,power_transitions[i][state[i]][1]) for i in range(len(state))]

            lookahead_transitions = sorted(lookahead_transitions,key=lambda k: k[1],reverse=True)

            if n_step != 0:
                indices = lookahead_transitions[:budget]
                indices = [i[0] for i in indices]
            else:
                indices = greedy_transitions 
                if len(indices) > budget:
                    indices = np.random.choice(indices,budget)
                elif len(indices) < budget:
                    others = [i[0] for i in lookahead_transitions if i[0] not in indices]
                    indices += others[:budget-len(indices)]

            action = np.zeros(N, dtype=np.int8)
            action[indices] = 1
            _, reward, done, _ = env.step(action)
            if done and t+1 < T: env.reset()
            all_reward[epoch, t] = reward

    return all_reward

def greedy_iterative_policy(env, n_episodes, n_epochs,discount,reward_function='matching',lamb=0,use_Q=False,use_whittle=False,use_shapley=False):
    """ random action each timestep """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    if reward_function != 'combined':
        raise Exception("Reward function is not matching; greedy is designed for match + activity")

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    def powerset(iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return itertools.chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()
        print('first state', env.observe())
        match_probabilities = np.array(env.match_probability_list)[env.cohort_idx]
        true_transitions = env.transitions

        if use_Q:
            Q_vals = arm_value_iteration_exponential(env.transitions, discount, budget, env.match_probability,reward_function=reward_function,lamb=lamb,match_probability_list=match_probabilities)

        reward_by_group = np.zeros((2**N,2**N))
        for temp_state in list(itertools.product([0, 1], repeat=N)):
            for bitstring in list(itertools.product([0, 1], repeat=N)):
                activity_score = 0
                non_match_prob = 1

                for i in range(len(bitstring)):
                    if bitstring[i] == 1:
                        activity_score = np.sum(temp_state[i])
                        non_match_prob *= (1-match_probabilities[i]*temp_state[i])
                reward_by_group[binary_to_decimal(temp_state),binary_to_decimal(bitstring)] = non_match_prob + lamb*activity_score

        for t in range(0, T):
            state = env.observe()

            selected_idx = []
            current_non_match_prob = 1

            match_probabilities = np.array(env.match_probability_list)[env.cohort_idx]

            for _ in range(budget):
                current_action = [0 for i in range(N)]
                for i in selected_idx:
                    current_action[i] = 1 
                encoded_state = binary_to_decimal(state)

                scores = []
                for i in range(N):
                    if i in selected_idx:
                        continue 
                    
                    if use_whittle:
                        arm_transitions = true_transitions[i, :, :, 1]
                        min_chosen_subsidy = -1
                        score = arm_compute_whittle(arm_transitions, state[i], discount, min_chosen_subsidy,reward_function=reward_function,lamb=lamb,match_prob=(1-current_non_match_prob*(1-match_probabilities[i])))
                    else:
                        score = 0
                        if use_shapley:
                            num = 0
                            non_selected = [j for j in range(N) if j not in selected_idx and j!=i]
                            for combo in powerset(non_selected):
                                indices = [0 for i in range(N)]
                                for j in combo:
                                    indices[j] = 1 
                                for j in selected_idx:
                                    indices[j] = 1 
                                initial_score = reward_by_group[binary_to_decimal(state)][binary_to_decimal(indices)]
                                indices[i] = 1
                                final_score = reward_by_group[binary_to_decimal(state)][binary_to_decimal(indices)]
                                num += 1
                                score += (final_score-initial_score)
                            score /= num 

                            if use_Q:
                                # subtract current reward, add Shapley
                                activity_score = np.sum(state)           
                                matching_score = current_non_match_prob - current_non_match_prob*(1-match_probabilities[i]*state[i])                            
                                score -= matching_score + activity_score * lamb

                        if use_Q:
                            new_action = deepcopy(current_action)
                            new_action[i] = 1

                            encoded_action = binary_to_decimal(new_action)
                            future_score = Q_vals[encoded_state,encoded_action]
                            score += future_score
                    scores.append((score,i))
                selected_idx.append(max(scores,key=lambda k: k[0])[1])
                current_non_match_prob *= (1-match_probabilities[selected_idx[-1]])
            # select arms at random
            selected_idx = np.array(selected_idx)
            action = np.zeros(N, dtype=np.int8)
            action[selected_idx] = 1

            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            all_reward[epoch, t] = reward

    return all_reward




def WIQL(env, n_episodes, n_epochs):
    """ Whittle index-based Q-Learning
    [Biswas et al. 2021]

    input: N, budget, alpha(c), initial states
    """
    N         = env.cohort_size
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes

    env.reset_all()

    all_reward = np.zeros((n_epochs, T))

    def alpha_func(c):
        """ learning parameter
        alpha = 0: agent doesn't learn anything new
        alpha = 1: stores only most recent information; overwrites previous records """
        assert 0 <= c <= 1
        return c

    for epoch in range(n_epochs):
        if epoch != 0: env.reset_instance()

        # initialize
        Q_vals    = np.zeros((N, n_states, n_actions))
        lamb_vals = np.zeros((N, n_states))

        print('first state', env.observe())
        for t in range(0, T):
            state = env.observe()

            # select M arms using epsilon-decay policy
            epsilon = N / (N + t)

            # with probability epsilon, select B arms uniformly at random
            if np.random.binomial(1, epsilon):
                selected_idx = np.random.choice(N, size=budget, replace=False)
            else:
                state_lamb_vals = np.array([lamb_vals[i, state[i]] for i in range(N)])
                # select top arms according to their lambda values
                selected_idx = np.argpartition(state_lamb_vals, -budget)[-budget:]
                selected_idx = selected_idx[np.argsort(state_lamb_vals[selected_idx])][::-1] # sort indices

            action = np.zeros(N, dtype=np.int8)
            action[selected_idx] = 1

            # take suitable actions on arms
            # execute chosen policy; observe reward and next state
            next_state, reward, done, _ = env.step(action)
            if done and t+1 < T: env.reset()

            # update Q, lambda
            c = .5 # None
            for i in range(N):
                for s in range(n_states):
                    for a in range(n_actions):
                        prev_Q = Q_vals[i, s, a]
                        state_i = next_state[i]
                        prev_max_Q = np.max(Q_vals[i, state_i, :])

                        alpha = alpha_func(c)

                        Q_vals[i, s, a] = (1 - alpha) * prev_Q + alpha * (reward + prev_max_Q)

                    lamb_vals[i, s] = Q_vals[i, s, 1] - Q_vals[i, s, 0]

            all_reward[epoch, t] = reward

    return all_reward

def mcts_greedy_policy(env,state,budget,lamb,memory,per_epoch_results):
    return run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts,VolunteerStateTwoStep,two_step_idx_to_action)

def arm_value_iteration_sufficient(transitions, state, T_stat,predicted_subsidy, discount, n_arms,match_prob,budget,threshold=value_iteration_threshold,reward_function='activity',lamb=0,probs=[]):
    """ value iteration for a single arm at a time

    value iteration for the MDP defined by transitions with lambda-adjusted reward function
    return action corresponding to pi^*(s_I)
    """
    assert discount < 1

    n_states, n_actions = transitions.shape
    p = match_prob

    value_func = np.zeros(n_arms+1,n_states)
    for i in range(value_func.shape[0]):
        for j in range(value_func.shape[1]):
            value_func[i,j] = random.random()
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
            return (1-np.power(1-p,T))/orig_T*(1-lamb) + lamb*s - a*predicted_subsidy 
            # return (1-np.power(1-p,T) - (1-np.power(1-p,T-1))) + lamb*s - a*predicted_subsidy
        elif T == 0 and a==s==1:
            return 0
        else:
            return s*a*(1-lamb) + lamb*s - a*predicted_subsidy 

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

        weights = np.array([random.random() for i in range(n_arms+1)])
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

        return (1-torch.pow(1-p,s.dot(a)))*(1-lamb) + lamb*torch.sum(s)/len(s)

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
            if random.random() < 0.05:
                action = [1 for i in range(budget)] + [0 for i in range(n_arms-budget)]
                binary_action = binary_to_decimal(action)
                random.shuffle(action)
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
                next_state.append(int(random.random()<one_probability))

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


def whittle_iterative(env,state,budget,lamb,memory, per_epoch_results):
    """Combination of the Whittle index + match probability
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and the Whittle memoizer"""


    N = len(state) 

    true_transitions = env.transitions

    if memory == None:
        memoizer = Memoizer('optimal')
    else:
        memoizer = memory 

    # TODO: Make this more general than \lamb = 0
    # Compute this for \lamb = 0 for now 

    people_to_add = set()

    if memory == None:
        memoizer = [Memoizer('optimal'),Memoizer('optimal')]
    else:
        memoizer = memory 
    
    # state_WI_activity, state_V_activity = whittle_v_index(env,state,budget,lamb,memoizer[0],reward_function="activity")
    state_WI_matching, state_V_matching, state_V_full_matching = whittle_v_index(env,state,budget,lamb,memoizer[1],reward_function="matching")

    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]

    true_transitions = env.transitions 
    max_probabilities = true_transitions[:,1,1,1]
    probable_future_value = 1

    for _ in range(budget):
        values = [0 for j in range(N)]
        previous_val = 1-np.prod([1-match_probabilities[j]*state[j] for j in list(people_to_add)])

        for i in range(N):
            if i not in people_to_add:
                current_val = match_probabilities[i]*state[i]
                # future_val = state_V_matching[i] + state_WI_matching[i]*max_probabilities[i]*(1/(1-env.discount)-1)
                # future_val -= current_val 
                # future_val *= env.discount
                #future_val = (state_WI_matching[i] - match_probabilities[i])/env.discount 
                future_val = state_V_full_matching[i,0]*true_transitions[i//env.volunteers_per_arm,state[i],1,0] + state_V_full_matching[i][1]*true_transitions[i//env.volunteers_per_arm,state[i],1,1]
                future_val -=  state_V_full_matching[i,0]*true_transitions[i//env.volunteers_per_arm,state[i],0,0] + state_V_full_matching[i][1]*true_transitions[i//env.volunteers_per_arm,state[i],0,1]
                future_val *= env.discount 

                future_match_prob = match_probabilities[i]*true_transitions[i//env.volunteers_per_arm,state[i],1,1]

                real_current_val = 1-np.prod([1-match_probabilities[j]*state[j] for j in list(people_to_add)])*(1-match_probabilities[i])
                ratio = (real_current_val - previous_val)/(match_probabilities[i]*state[i])
                ratio_future = (1-probable_future_value*(1-future_match_prob) - (1-probable_future_value))/(future_match_prob)
                if match_probabilities[i]*state[i] == 0:
                    ratio = 0
                if future_match_prob == 0:
                    ratio_future = 0 
                total_val = future_val*ratio_future + current_val*ratio  
                #total_val = future_val*ratio_future + current_val*ratio 
                values[i] = total_val 
        
        if np.max(values) > 0:
            idx = np.argmax(values)
            people_to_add.add(idx)
            probable_future_value *= (1-match_probabilities[idx]*true_transitions[
                idx//env.volunteers_per_arm,state[idx],1,1])
        else:
            break 

    people_to_add = list(people_to_add)

    action = np.zeros(N, dtype=np.int8)
    action[people_to_add] = 1

    return action, memoizer 
