import numpy as np
import heapq

from rmab.uc_whittle import Memoizer
from rmab.compute_whittle import arm_compute_whittle, arm_value_iteration_exponential, arm_value_iteration_neural, arm_compute_whittle_sufficient
from rmab.utils import get_stationary_distribution, binary_to_decimal, list_to_binary
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

class VolunteerState():
    def __init__(self):
        self.non_match_prob = 1
        self.previous_pulls = set()
        self.N = 0
        self.arm_values = []
        self.budget = 0
        self.use_Q = False 
        self.Q_vals = [] 
        self.state = []
        self.discount = 0.9
        self.lamb = 0

    def getPossibleActions(self):
        possibleActions = []
        for i in range(self.N):
            if i not in self.previous_pulls:
                possibleActions.append(Action(arm=i))
        return possibleActions

    def takeAction(self, action):
        newState = deepcopy(self)
        newState.non_match_prob *= (1-self.arm_values[action.arm])
        newState.previous_pulls.add(action.arm)
        return newState

    def isTerminal(self):
        if len(self.previous_pulls) == self.budget:
            return True 
        return False 

    def getReward(self):
        self.state_encoded = binary_to_decimal(self.state)
        self.action_encoded = binary_to_decimal(
            [0 if i not in self.previous_pulls else 1 for i in range(self.N)])

        value = 1-self.non_match_prob  + self.lamb*np.sum(self.state)

        if self.use_Q:
            value = self.Q_vals[
            self.state_encoded,self.action_encoded]
    
        return value 


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


def mcts_policy(env,state,budget,lamb,memory,per_epoch_results):
    match_probabilities = np.array(env.match_probability_list)[env.agent_idx]

    arm_values = [match_probabilities[i] for i in range(N)]
    initialState = VolunteerState()
    initialState.N = len(arm_values)
    initialState.arm_values = arm_values 
    initialState.budget = budget 
    initialState.lamb = lamb
    
    searcher = mcts(iterationLimit=100)
    selected_idx = []
    current_state = initialState 

    for i in range(budget):
        action = searcher.search(initialState=current_state)
        current_state = current_state.takeAction(action)
        selected_idx.append(action.arm)
    
    selected_idx = np.array(selected_idx)
    action = np.zeros(len(state), dtype=np.int8)
    action[selected_idx] = 1
    return action 


