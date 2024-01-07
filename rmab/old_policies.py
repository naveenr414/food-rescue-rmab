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
