import numpy as np
import random 

from rmab.whittle_policies import shapley_index_custom
from rmab.baseline_policies import random_policy, compute_p_matrix, compute_reward_matrix
from rmab.utils import contextual_custom_reward
from rmab.compute_whittle import Q_multi_prob, fast_compute_whittle_indices

import time

def get_reward(state,action,match_probs,lamb):
    prod_state = 1-state*action*np.array(match_probs)
    prob_all_inactive = np.prod(prod_state)
    return (1-prob_all_inactive)*(1-lamb) + np.sum(state)/len(state)*lamb

def get_reward_max(state,action,match_probs,lamb):
    prod_state = state*action*np.array(match_probs)
    score = np.max(prod_state)
    return score*(1-lamb) + np.sum(state)/len(state)*lamb

def get_reward_custom(state,action,match_probs,lamb,reward_type,reward_parameters,active_states,context):
    return contextual_custom_reward(state,action,match_probs,reward_type,reward_parameters,active_states,context)*(1-lamb) + np.sum(state)/len(state)*lamb

class MonteCarloTreeSearchNode():
    """Class which allows for MCTS to run
    Does so by abstracting away reward specifics
    To State-Action nodes"""

    def __init__(self, state, simulation_no,transitions,parent=None, parent_action=None,use_whittle=False,memoizer=None,time_limit=100):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = []
        self.results_children = {}
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        self.transitions = transitions 
        self.use_whittle = use_whittle
        self.memoizer = memoizer
        self.simulation_no = simulation_no
        self.use_max = False 
        self.time_limit = time_limit 

    def untried_actions(self):
        self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    def q(self):
        """Get the Q Value for a Node
        
        At the leaf level, we take the average over all runs
        When using Whittle indices, this isn't necesary, but doesn't hurt
        At a non-leaf level, we find the best possible leaf node, and let it's reward
        Be our reward
        
        Returns: Float, Q value for node"""

        if self.is_terminal_node():
            return np.mean(self._results)*self._number_of_visits
        else:
            if self.use_max:
                return np.max(list(self.results_children.values()))*self._number_of_visits
            else:
                if self._number_of_visits == 0:
                    return 0
                return np.mean(self._results)*self._number_of_visits

    def n(self):
        return self._number_of_visits

    def expand(self):
        """Expand the current node and determine which child node to pursue
        
        Arguments: None 
        
        Returns: New Child Node that's being expanded"""

        action = self._untried_actions.pop()
        next_state = StateAction.move(self.state,action)
        next_state.memory = self.state.memory 
        child_node = MonteCarloTreeSearchNode(
            next_state, self.simulation_no,self.transitions,parent=self, parent_action=action,use_whittle=self.use_whittle)

        self.children.append(child_node)
        return child_node 

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        """Use either a random or whittle-based policy to rollout 
            Un-selected arms
        
        Arguments: None
        
        Returns: Reward/result from the current rollout"""

        current_rollout_state = self.state

        if not self.state.use_raw_reward:
            game_rewards = []
            while not current_rollout_state.is_game_over():
                game_rewards.append(current_rollout_state.game_result())
                possible_moves = current_rollout_state.get_legal_actions()
                action = self.rollout_policy(possible_moves)
                current_rollout_state = current_rollout_state.move(action) 
            game_rewards.append(current_rollout_state.game_result())
            return max(game_rewards,key=lambda k: k[1])

        else:
            while not current_rollout_state.is_game_over():
                possible_moves = current_rollout_state.get_legal_actions()
                action = self.rollout_policy(possible_moves)
                current_rollout_state = current_rollout_state.move(action) 
            return current_rollout_state.game_result()

    def backpropagate(self, result,action=None):
        """Backpropogation: Update parent nodes with the child node's reward
            For terminal nodes, maintain a list of rewards seen so far
            For non-terminal nodes, maintain the maximum over all children
        
        Returns: Nothing 
        
        Side Effects: Updates all seen results across parent nodes"""

        self._number_of_visits += 1.
        if self.is_terminal_node():
            self._results.append(result)
            result = np.mean(self._results)
        else:
            if self.use_max:
                if action in self.results_children:
                    self.results_children[action] = max(result,self.results_children[action])
                else:
                    self.results_children[action] = result 
            else:
                self._results.append(result)
        if self.parent:
            self.parent.backpropagate(result,self.parent_action)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=5):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) if c.n() > 0 else 0 for c in self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):  
        if self.use_whittle and random.random() < 0.9:
            state_WI = self.state.whittle_index 

            relevant_WI = [state_WI[i] for i in possible_moves]
            best_move = possible_moves[np.argmax(relevant_WI)]
            return best_move 
        return possible_moves[random.randint(0,len(possible_moves)-1)]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self,budget):
        """Find the best set of arms to pull
        Run a set of simulations, then find the best arm-combo
        
        Arguments: Float, budget, maximum number of arms to pull
        
        Returns: List of pulled arms, of length budget"""

        start = time.time() 

        simulation_no = self.simulation_no
        best_action = []
        best_reward = 0
        for i in range(simulation_no):
            v = self._tree_policy()
            last_action, reward = v.rollout()

            if reward > best_reward:
                best_reward = reward 
                best_action = last_action 

            if time.time()-start > self.time_limit:
                break 
        curr_node = self 

        if self.state.use_raw_reward:
            actions = []
            for i in range(budget):
                if len(curr_node.children) == 0:
                    possible_values = curr_node.state.get_legal_actions()
                    actions += random.sample(possible_values,budget-i)
                    return actions 
                else:
                    best_child = curr_node.best_child(c_param=0.)
                    actions.append(best_child.parent_action)
                    curr_node = best_child 
            return actions 
        else:
            return [i for i in range(len(best_action)) if best_action[i] == 1]
    
    def __str__(self):
        return str(self.state)

class StateAction():
    """Class which abstracts out the game specifics
    Covers the reward and updates the arms pulled so far"""

    def __init__(self,budget,discount,lamb,initial_state,volunteers_per_arm,n_arms,match_probs,max_rollout_actions,env,shapley=False,use_raw_reward=False,p_matrix=None,contextual=True):
        self.budget = budget 
        self.discount = discount 
        self.lamb = lamb  
        self.volunteers_per_arm = volunteers_per_arm 
        self.n_arms = n_arms
        self.match_probs = match_probs 
        self.max_rollout_actions = max_rollout_actions 
        self.previous_state_actions = []
        self.current_state = initial_state 
        self.env = env 
        self.shapley = shapley 
        self.memory = None
        self.attribution_method = "proportional"
        self.use_raw_reward = use_raw_reward
        self.whittle_index = []
        self.p_matrix = p_matrix
        self.contextual = contextual

    def get_legal_actions(self):
        """Find all the arms pulled, and make sure no duplicate arms are pulled
        
        Arguments: None
        
        Returns: List of arms that are pullable"""

        idx = len(self.previous_state_actions)//self.budget*self.budget 
        current_state_actions = self.previous_state_actions[idx:]
        taken_actions = set([i[1] for i in current_state_actions])
        all_actions = set(range(self.volunteers_per_arm * self.n_arms))
        valid_actions = list(all_actions.difference(taken_actions))

        return valid_actions 
    
    def is_game_over(self):
        return len(self.previous_state_actions) >= self.max_rollout_actions 
    
    def game_result(self):
        """Find the modified Whittle Index when playing self.previous_state_actions
            First, determine what actions are played in the current state
            Compute the corresponding rewards now and later using this
            Then, compute the best arms to pull through a modified Whittle index
            The corresponding objective value is the sum of the Q values
        
        Arguments: None
        
        Returns: Modified Whittle Index; we know our current reward, 
            And we know the future potential rewards
            Use this to compute the Whittle Index"""

        last_state = []

        # Compute the State, Arms Played, and immideate marginal rewards
        state_choices = self.previous_state_actions
        corresponding_actions = [i[1] for i in state_choices]
        last_state = np.array(state_choices[0][0])
        last_action = []
        for arm in range(len(last_state)):
            if arm in corresponding_actions:
                last_action.append(1)
            else:
                last_action.append(0)
        last_action = np.array(last_action)

        if self.contextual:
            last_reward = get_reward_custom(last_state,last_action,self.match_probs,self.lamb,self.env.reward_type,self.env.reward_parameters,self.env.active_states,self.env.context)
        else:
            last_reward = get_reward_custom(last_state,last_action,self.match_probs,self.lamb,self.env.reward_type,self.env.reward_parameters,self.env.active_states,np.array(self.env.match_probability_list)[self.env.agent_idx])
        last_reward -= np.sum(last_state)/len(last_state)*self.lamb 


        if self.use_raw_reward:
            total_reward = 0
            for i in range(self.max_rollout_actions//self.budget):
                state_choices = self.previous_state_actions[i*self.budget:(i+1)*self.budget]
                corresponding_actions = [i[1] for i in state_choices]
                corresponding_state = np.array(state_choices[0][0])
                action_0_1 = []
                for arm in range(len(corresponding_state)):
                    if arm in corresponding_actions:
                        action_0_1.append(1)
                    else:
                        action_0_1.append(0)
                action_0_1 = np.array(action_0_1)

                total_reward += self.discount**i * get_reward_custom(corresponding_state,action_0_1,self.match_probs,self.lamb,self.env.reward_type,self.env.reward_parameters,self.env.active_states,self.env.context)
            return last_action, total_reward 
        arm_q = 0

        for i in range(len(last_state)):
            arm_q += self.memory[0][i][last_state[i]][last_action[i]]

        return last_action, last_reward + arm_q         
    
    def move(initial_state,action):
        """Play an action, and create a new StateAction from that
        
        Arguments: 
            initial_state: Object from StateAction
            action: Integer, which arm is being pulled
        
        Returns: New StateAction resulting from the pulled arm"""

        previous_state_actions = initial_state.previous_state_actions + [(initial_state.current_state,action)]

        new_state_object = StateAction(initial_state.budget,initial_state.discount,initial_state.lamb,initial_state.current_state,initial_state.volunteers_per_arm,initial_state.n_arms,initial_state.match_probs,initial_state.max_rollout_actions,initial_state.env,p_matrix=initial_state.p_matrix)
        new_state_object.use_raw_reward = initial_state.use_raw_reward
        new_state_object.attribution_method = initial_state.attribution_method
        new_state_object.previous_state_actions = previous_state_actions
        new_state_object.shapley = initial_state.shapley 
        new_state_object.whittle_index = initial_state.whittle_index 
        new_state_object.memory = initial_state.memory 
        new_state_object.contextual = initial_state.contextual 

        if len(previous_state_actions)%new_state_object.budget == 0 and initial_state.max_rollout_actions > initial_state.budget:
            new_state = [0 for i in range(len(new_state_object.current_state))]
            played_actions = [i[1] for i in previous_state_actions[-new_state_object.budget:]]
            action_0_1 = [0 for i in range(len(new_state))]

            for i in played_actions:
                action_0_1[i] = 1

            for i in range(new_state_object.n_arms):
                for j in range(new_state_object.volunteers_per_arm):
                    idx = i*new_state_object.volunteers_per_arm + j
                    prob = initial_state.env.transitions[i, new_state_object.current_state[idx], action_0_1[idx], :]
                    next_state = int(random.random()>prob[0])
                    new_state[idx] = next_state
            new_state_object.current_state = new_state

        return new_state_object 
    
    def __str__(self):
        return str(self.previous_state_actions)


def mcts_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",use_whittle=False,contextual=True):
    """MCTS Policy which computes the best arms to pull
    Considers a rollout depth of budget*some constant
    
    Arguments: 
        env: Simulator Environment
        state: Num Agents x 2 numpy array (0-1)
        budget: Integer, how many arms we can pull
        lamb: Balance between engagement, global reward
        Memory: Contains the Shapley values, a memoizer 
        per_epoch_results: Optional argument, nothing for this 
    
    Returns: Action, 0-1 list"""
    
    if env.is_training:
        return random_policy(env,state,budget,lamb,memory, per_epoch_results)
    N = len(state)
    rollout = budget*env.mcts_depth

    match_probs = np.array(env.match_probability_list)[env.agent_idx]
    state_actions = []
    if memory == None:
        memory = None,np.array(shapley_index_custom(env,np.ones(len(env.agent_idx)),{})[0])

    s = StateAction(budget,env.discount,lamb,state,env.volunteers_per_arm,env.cohort_size,match_probs,rollout,env,use_raw_reward=True)
    s.contextual=contextual
    s.memory = memory 
    root = MonteCarloTreeSearchNode(s,env.mcts_test_iterations,transitions=env.transitions,use_whittle=False)
    selected_idx = root.best_action(budget)
    memory = s.memory 
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, memory 

def run_mcts(env,Q_multi_prob_list, p_matrix,whittle_matrix,budget,state,lamb,contextual=True):
    """Boilerplate for running the MCTS method
    
    Arguments: 
        env: Simulator Environment
        Q_multi_prob_list: Matrix of Q values for each agent x state
        p_matrix: Matrix of marginal rewards for each agent x state
        whittle_matrix: Numpy matrix of size N x States, which captures Whittle index for each
            combination
        budget: Integer, how many arms we can pull
        state: Numpy array, state for each arm
        lamb: Balance between engagement, global reward
        contextual: Boolean, is this a contextual or non-contextual experiment
    
    Returns: Action, 0-1 list"""
  
    
    start = time.time() 
    N = len(state)
    s = StateAction(budget,env.discount,lamb,state,env.volunteers_per_arm,env.cohort_size,env.match_probability_list[env.agent_idx],budget,env,shapley=False,p_matrix=p_matrix)
    s.contextual=contextual
    s.attribution_method = "proportional" 
    s.previous_state_actions = []
    s.memory = (Q_multi_prob_list,p_matrix) 
    s.per_epoch_results = None
    s.whittle_index = [whittle_matrix[i][state[i]] for i in range(N)]
    root = MonteCarloTreeSearchNode(s,env.mcts_test_iterations,transitions=env.transitions,time_limit=env.time_limit,use_whittle=True)
    selected_idx = root.best_action(budget)
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, (Q_multi_prob_list, p_matrix,whittle_matrix)


def mcts_linear_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",attribution_method="proportional",contextual=True):
    """Leverage Shapley values + MCTS to compute indices 
    Basically uses MCTS to compute the potential rewards immideatly
    Then uses Shapley indices to estimate future rewards
    This leads to index computation
    
    Arguments: 
        env: Simulator Environment
        state: Num Agents x 2 numpy array (0-1)
        budget: Integer, how many arms we can pull
        Lamb: Balance between engagement, global reward
        Memory: Contains the V, Pi network
        per_epoch_results: Optional argument, nothing for this 

    Returns: 0-1 list of arms to pull"""
    
    N = len(state)
    n_states = env.transitions.shape[1] 

    if memory == None:
        start = time.time()
        p_matrix = compute_p_matrix(env,N) 
        Q_multi_prob_list = np.zeros((N,n_states,2))

        for i in range(N):
            for s in range(n_states):
                Q_multi_prob_list[i][s] = Q_multi_prob(env.transitions[i//env.volunteers_per_arm], s, 0,env.discount,lamb=lamb,
                match_prob=0,match_prob_now=0,num_arms=len(state),active_states=env.active_states,
                p_matrix=p_matrix[i])[-1]

        reward_matrix = compute_reward_matrix(env,N,lamb)
        reward_matrix[:,:,1] += (1-lamb)*p_matrix

        whittle_matrix = np.zeros((N,n_states))
        for i in range(N):
            for j in range(n_states):
                whittle_matrix[i] = fast_compute_whittle_indices(env.transitions[i//env.volunteers_per_arm],reward_matrix[i],env.discount)
    else:
        Q_multi_prob_list, p_matrix, whittle_matrix = memory 

    return run_mcts(env,Q_multi_prob_list,p_matrix,whittle_matrix,budget,state,lamb,contextual=contextual)

def non_contextual_mcts_linear_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",attribution_method="proportional"):
    """Run a variant of the MCTS Linear Policy, where contexts aren't used"""
    
    return mcts_linear_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup=group_setup,attribution_method=attribution_method,contextual=False)

def mcts_shapley_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",attribution_method="proportional",contextual=True):
    """Leverage Shapley values + MCTS to compute indices + rollout 
    Basically uses MCTS to compute the potential rewards immideatly
    Then uses Shapley indices to estimate future rewards
    This leads to index computation
    
    Arguments: 
        env: Simulator Environment
        state: Num Agents x 2 numpy array (0-1)
        budget: Integer, how many arms we can pull
        Lamb: Balance between engagement, global reward
        Memory: Contains the V, Pi network
        per_epoch_results: Optional argument, nothing for this 

    Returns: 0-1 list of arms to pull"""
    
    N = len(state)
    n_states = env.transitions.shape[1] 
    rollout = budget
    state_actions = []

    if memory == None:
        u_matrix = np.zeros((N,n_states))

        for s in range(n_states):
            for i in range(N):
                curr_state = [env.best_state for _ in range(N)]
                curr_state[i] = s
                u_matrix[i,s] = shapley_index_custom(env,curr_state,{},idx=i)

        Q_multi_prob_list = np.zeros((N,n_states,2))

        for i in range(N):
            for s in range(n_states):
                Q_multi_prob_list[i][s] = Q_multi_prob(env.transitions[i//env.volunteers_per_arm], s, 0,env.discount,lamb=lamb,
                match_prob=0,match_prob_now=0,num_arms=len(state),active_states=env.active_states,
                p_matrix=u_matrix[i])[-1]

        reward_matrix = np.zeros((N,n_states,2))

        for i in range(N):
            for j in range(n_states):
                if j in env.active_states:
                    reward_matrix[i,j] += lamb/N
                reward_matrix[i,j,1] += (1-lamb)*u_matrix[i,j]

        whittle_matrix = np.zeros((N,n_states))
        for i in range(N):
            for j in range(n_states):
                whittle_matrix[i] = fast_compute_whittle_indices(env.transitions[i//env.volunteers_per_arm],reward_matrix[i],env.discount)

    else:
        Q_multi_prob_list, u_matrix, whittle_matrix = memory 

    return run_mcts(env,Q_multi_prob_list,u_matrix,whittle_matrix,budget,state,lamb,contextual=contextual)

def non_contextual_mcts_shapley_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup="none",attribution_method="proportional"):
    """Run a variant of the MCTS Shapley policy where it doesn't incorporate the current context"""
    
    return mcts_shapley_policy(env,state,budget,lamb,memory,per_epoch_results,group_setup=group_setup,attribution_method=attribution_method,contextual=False)