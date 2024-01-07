import numpy as np
import heapq

from rmab.uc_whittle import Memoizer
from rmab.compute_whittle import arm_compute_whittle, arm_value_iteration_exponential, arm_value_iteration_neural, arm_compute_whittle_sufficient
from rmab.utils import get_stationary_distribution, binary_to_decimal, list_to_binary
from rmab.omniscient_policies import greedy_policy, whittle_greedy_policy, whittle_index, shapley_whittle_policy, whittle_policy
from itertools import combinations
from rmab.compute_whittle import get_q_vals

import torch
import torch.nn as nn
import torch.optim as optim
import itertools 
from openrl.modules.common import PPONet as Net
from openrl.runners.common import PPOAgent as Agent
from scipy.stats import binom
from copy import deepcopy
from mcts import mcts, randomPolicy, treeNode
import random 
import math 
import time 

def findMaxChild(root,explorationValue):
    bestVal = 0

    if root.state.isTerminal():
        return root.totalReward / root.numVisits + explorationValue * math.sqrt(
        2 * math.log(root.parent.numVisits) / root.numVisits)

    for child in root.children.values():
        bestVal = max(bestVal,findMaxChild(child,explorationValue))

    return bestVal 


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

    def get_current_reward(self,state,previous_pulls):
        policy = 'index'


        if policy == 'reward':
            non_match_prob = 1 
            for i in previous_pulls:
                expected_active_rate = 0
                for i in range(len(state)):
                    action = int(i in previous_pulls)
                    expected_active_rate += self.transitions[i//self.volunteers_per_arm,state[i],action,1]

                non_match_prob *= (1-state[i]*self.arm_values[i])

            value = 1-non_match_prob  + self.lamb*expected_active_rate

        elif policy == 'index':
            state_WI = np.zeros(len(state))

            # TODO: Incorporate the Q version of WI back in 
            # for i in range(len(state_WI)):
            #     arm_transitions = self.env.transitions[i//self.volunteers_per_arm, :, :, 1]
            #     check_set_val = self.memoizer.check_set(arm_transitions, state[i])

            #     if check_set_val != -1:
            #         state_WI[i] = check_set_val
                    
            #     else:
            #         state_WI[i] = get_q_vals(arm_transitions, state[i], 0, self.discount,reward_function="activity",lamb=self.lamb)[1]
            #         self.memoizer.add_set(arm_transitions, state[i], state_WI[i])

            state_WI = whittle_index(self.env,state,self.budget,self.lamb,self.memoizer,reward_function="activity")
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

        action = [int(i in self.previous_pulls) for i in range(len(self.states))]

        for i in range(self.max_depth-1):
            next_states = np.zeros(self.cohort_size*self.volunteers_per_arm,dtype=int)
            for i in range(self.cohort_size):
                for j in range(self.volunteers_per_arm):
                    idx = i*self.volunteers_per_arm + j
                    prob = self.transitions[i, self.states[idx], action[idx], :]
                    next_state = np.random.choice(a=2, p=prob)
                    next_states[idx] = next_state
            
            if ''.join([str(i) for i in next_states]) in self.memory:
                action = self.memory[''.join([str(i) for i in next_states])]
            else:
                action = [1] * self.budget + [0] * (len(self.states) - self.budget)
                random.shuffle(action)

            reward = self.get_current_reward(next_states, {index for index, value in enumerate(action) if value == 1})
            previous_rewards.append(reward)

        for i in previous_rewards:
            total_value += disc*i 
            disc *= self.discount 

        return total_value 
    def set_state(self,states):
        self.states = states

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

class VolunteerStateTwoStepMCTS():
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

        self.num_by_cohort = []
        self.current_cohort = 0
        self.num_in_cohort = 0


    def set_num_by_cohort(self,num_by_cohort):
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

    def isTerminal(self):
        if len(self.previous_pulls) == self.budget:
            return True 
        return False 

    def get_current_reward(self,state,previous_pulls):
        policy = 'index'


        if policy == 'reward':
            non_match_prob = 1 
            for i in previous_pulls:
                expected_active_rate = 0
                for i in range(len(state)):
                    action = int(i in previous_pulls)
                    expected_active_rate += self.transitions[i//self.volunteers_per_arm,state[i],action,1]

                non_match_prob *= (1-state[i]*self.arm_values[i])

            value = 1-non_match_prob  + self.lamb*expected_active_rate

        elif policy == 'index':
            state_WI = np.zeros(len(state))

            # TODO: Incorporate the Q version of WI back in 
            # for i in range(len(state_WI)):
            #     arm_transitions = self.env.transitions[i//self.volunteers_per_arm, :, :, 1]
            #     check_set_val = self.memoizer.check_set(arm_transitions, state[i])

            #     if check_set_val != -1:
            #         state_WI[i] = check_set_val
                    
            #     else:
            #         state_WI[i] = get_q_vals(arm_transitions, state[i], 0, self.discount,reward_function="activity",lamb=self.lamb)[1]
            #         self.memoizer.add_set(arm_transitions, state[i], state_WI[i])

            state_WI = whittle_index(self.env,state,self.budget,self.lamb,self.memoizer,reward_function="activity")
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

        action = [int(i in self.previous_pulls) for i in range(len(self.states))]

        for i in range(self.max_depth-1):
            next_states = np.zeros(self.cohort_size*self.volunteers_per_arm,dtype=int)
            for i in range(self.cohort_size):
                for j in range(self.volunteers_per_arm):
                    idx = i*self.volunteers_per_arm + j
                    prob = self.transitions[i, self.states[idx], action[idx], :]
                    next_state = np.random.choice(a=2, p=prob)
                    next_states[idx] = next_state
            
            if ''.join([str(i) for i in next_states]) in self.memory:
                action = self.memory[''.join([str(i) for i in next_states])]
            else:
                action = [1] * self.budget + [0] * (len(self.states) - self.budget)
                random.shuffle(action)

            reward = self.get_current_reward(next_states, {index for index, value in enumerate(action) if value == 1})
            previous_rewards.append(reward)

        for i in previous_rewards:
            total_value += disc*i 
            disc *= self.discount 

        return total_value 
    def set_state(self,states):
        self.states = states



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

def one_step_idx_to_action(selected_idx,env,state,lamb,memory):
    selected_idx = np.array(selected_idx)
    action = np.zeros(len(state), dtype=np.int8)
    action[selected_idx] = 1

    return action, memory

def two_step_idx_to_action(selected_idx,env,state,lamb,memory):
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

def two_step_mcts_to_action(selected_idx,env,state,lamb, memory): 
    # TODO: Replace time limits with iteration limits
    fractions = [1/3,1/2]
    fractions.append((1-sum(fractions))/2)

    greedy_action, _ = mcts_policy(env,state,env.budget,lamb,memory,None,timeLimit=env.TIME_PER_RUN * fractions[0])
    bin_counts = np.zeros(env.cohort_size, dtype=int)

    for i in range(env.cohort_size):
        bin_counts[i] = np.sum(greedy_action[i*env.volunteers_per_arm:(i+1)*env.volunteers_per_arm])

    num_by_cohort = bin_counts

    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    N = len(state)

    arm_values = [match_probabilities[i] for i in range(N)]
    initialState = VolunteerStateTwoStepMCTS()
    initialState.N = len(arm_values)
    initialState.arm_values = arm_values 
    initialState.budget = env.budget 
    initialState.lamb = lamb
    initialState.transitions = env.transitions
    initialState.cohort_size = env.cohort_size
    initialState.volunteers_per_arm = env.volunteers_per_arm
    initialState.set_state(state)
    initialState.discount = env.discount 
    initialState.memory = memory[0]
    initialState.memoizer = memory[1]
    initialState.env = env 

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

def two_step_mcts_to_whittle(selected_idx,env,state,lamb, memory): 
    # TODO: Replace time limits with iteration limits
    fractions = [1/2]
    fractions.append((1-sum(fractions))/2)

    greedy_action, _ = whittle_greedy_policy(env,state,env.budget,lamb,memory[1],None)
    bin_counts = np.zeros(env.cohort_size, dtype=int)
    
    for i in range(env.cohort_size):
        bin_counts[i] = np.sum(greedy_action[i*env.volunteers_per_arm:(i+1)*env.volunteers_per_arm])

    num_by_cohort = bin_counts

    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
    N = len(state)

    arm_values = [match_probabilities[i] for i in range(N)]
    initialState = VolunteerStateTwoStepMCTS()
    initialState.N = len(arm_values)
    initialState.arm_values = arm_values 
    initialState.budget = env.budget 
    initialState.lamb = lamb
    initialState.transitions = env.transitions
    initialState.cohort_size = env.cohort_size
    initialState.volunteers_per_arm = env.volunteers_per_arm
    initialState.set_state(state)
    initialState.discount = env.discount 
    initialState.memory = memory[0]
    initialState.memoizer = memory[1]
    initialState.env = env 

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


def run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts_function,volunteer_class,idx_to_action,num_iterations=100,timeLimit=-1):
    """Compute an MCTS-based policy which selects arms to notify 
    sequentially, then rolls out for Q=5 steps to predict reward

    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Information on previously computed Whittle indices, the memoizer
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    if memory == None:
        memory = [{}, Memoizer('optimal')]

    if ''.join([str(i) for i in state]) in memory[0]:
        return memory[0][''.join([str(i) for i in state])], memory
    
    if num_iterations > 0:
        match_probabilities = np.array(env.match_probability_list)[env.agent_idx]
        N = len(state)

        arm_values = [match_probabilities[i]for i in range(N)]
        initialState = volunteer_class()
        initialState.N = len(arm_values)
        initialState.arm_values = arm_values 
        initialState.budget = budget 
        initialState.lamb = lamb
        initialState.transitions = env.transitions
        initialState.cohort_size = env.cohort_size
        initialState.volunteers_per_arm = env.volunteers_per_arm
        initialState.set_state(state)
        initialState.discount = env.discount 
        initialState.memory = memory[0]
        initialState.memoizer = memory[1]
        initialState.env = env 
        

        # TODO: Switch back to iterationLimit 
        #searcher = mcts_function(iterationLimit=num_iterations)
        if timeLimit != -1:
            searcher = mcts_function(timeLimit=timeLimit*4/5)
        else:
            searcher = mcts_function(timeLimit=env.TIME_PER_RUN*4/5)
        
        selected_idx = []
        current_state = initialState 

        for _ in range(budget):
            action = searcher.search(initialState=current_state)
            current_state = current_state.takeAction(action)
            selected_idx.append(action.arm)

            if timeLimit != -1:
                searcher = mcts_function(timeLimit=timeLimit*1/10)
            else:
                searcher = mcts_function(timeLimit=env.TIME_PER_RUN*1/10)


        memory[1] = initialState.memoizer

    else:
        selected_idx = []
    action, memory = idx_to_action(selected_idx,env,state,lamb,memory)

    memory[0][''.join([str(i) for i in state])] = action 
    return action, memory


def mcts_policy(env,state,budget,lamb,memory,per_epoch_results,num_iterations=100,timeLimit=-1):
    return run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts,VolunteerState,one_step_idx_to_action,num_iterations=num_iterations,timeLimit=timeLimit)

def mcts_max_policy(env,state,budget,lamb,memory,per_epoch_results):
    return run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts_max,VolunteerState,one_step_idx_to_action)

def mcts_greedy_policy(env,state,budget,lamb,memory,per_epoch_results):
    return run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts,VolunteerStateTwoStep,two_step_idx_to_action)

def mcts_mcts_policy(env,state,budget,lamb,memory,per_epoch_results):
    return run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts,VolunteerStateTwoStep,two_step_mcts_to_action,num_iterations=0)

def mcts_whittle_policy(env,state,budget,lamb,memory,per_epoch_results):
    return run_mcts(env,state,budget,lamb,memory,per_epoch_results,mcts,VolunteerStateTwoStep,two_step_mcts_to_whittle,num_iterations=0)
