import gym
import numpy as np
import gymnasium.spaces as spaces
from rmab.fr_dynamics import get_db_data, train_rf,  get_match_probabilities
from rmab.utils import custom_reward
import random
import torch 
import time 

class RMABSimulator(gym.Env):
    '''
    This simulator simulates the interaction with a set of arms with unknown transition probabilities
    but with additional side information. This setup is aligned with restless multi-armed bandit problems
    where we do not have repeated access to the same set of arms, but instead a set of new arms may
    arrive in the next iteration with side information transferable between different iiterations.

    The inputs of the simulator are listed below:

        all_population: the total number of arms in the entire population
        all_features: this is a numpy array with shape (all_population, feature_size)
        all_transitions: this is a numpy array with shape (all_population, 2, 2)
                        state (NE, E), action (NI, I), next state (E)
        cohort_size: the number of arms arrive per iteration as a cohort
        episode_len: the total number of time steps per episode iteration
        budget: the number of arms that can be pulled in a time step


    This simulation supports two different setting of the features:
        - When features are multi-dimensional, the problem is a restless multi-armed bandit problem with
        side information.
        - When features are just single-dimensional with discrete values, the problem is a restless
        multi-armed bandit problem with group information.
        - In the extreme case where the group information is completely disjoint, it is the same as a
        restless multi-armed bandit problem with no information.

    '''

    def __init__(self, all_population, all_features, all_transitions, cohort_size, volunteers_per_arm,episode_len, n_instances, n_episodes, budget,
            discount,number_states=2,reward_style='state',match_probability=0.5,match_probability_list = [],TIME_PER_RUN=10.0,contextual=False):
        '''
        Initialization
        '''

        self.all_population  = all_population
        self.all_features    = all_features
        self.all_transitions = all_transitions
        self.cohort_size     = cohort_size
        self.volunteers_per_arm = volunteers_per_arm
        self.budget          = budget
        self.number_states   = number_states
        self.episode_len     = episode_len
        self.discount = discount
        self.n_episodes      = n_episodes   # total number of episodes per epoch
        self.n_instances     = n_instances  # n_epochs: number of separate transitions / instances
        self.reward_style = reward_style # Should we get reward style based on states or matches
        self.match_probability_list = match_probability_list
        self.TIME_PER_RUN = TIME_PER_RUN
        self.contextual = contextual 
        self.test_epochs = 0
        self.train_epochs = 0
        self.power = None # For the submodular runs 
        self.avg_reward = 5
        self.reward_type = "probability"
        self.reward_parameters = {}

        self.match_probability_list = np.array(self.match_probability_list)

        assert_valid_transition(all_transitions)

        # set up options for the multiple instances
        # track the first random initial state
        self.instance_count = 0
        self.episode_count  = 0
        self.timestep       = 0
        self.total_active = 0
        self.context_dim = 15

        # track indices of cohort members
        self.cohort_selection  = np.zeros((n_instances, cohort_size)).astype(int)
        self.first_init_states = np.zeros((n_instances, n_episodes, cohort_size*volunteers_per_arm)).astype(int)
        for i in range(n_instances):
            self.cohort_selection[i, :] = np.random.choice(a=self.all_population, size=self.cohort_size, replace=False)
            print('cohort', self.cohort_selection[i, :])
            for ep in range(n_episodes):
                self.first_init_states[i, ep, :] = self.sample_initial_states(self.cohort_size*self.volunteers_per_arm,prob=0.5)

        if contextual: 
            if len(self.match_probability_list) > 0:
                self.all_match_probabilities = self.match_probability_list
            else:
                donation_id_to_latlon, recipient_location_to_latlon, rescues_by_user, all_rescue_data, user_id_to_latlon = get_db_data()
                rf_classifier, evaluation = train_rf()
                self.all_match_probabilities = np.zeros((n_instances,n_episodes*episode_len,100 * self.volunteers_per_arm))
                self.all_context_features = np.zeros((n_instances,n_episodes*episode_len,100 * self.volunteers_per_arm,15))

                for i in range(n_instances):
                    self.all_match_probabilities[i], self.all_context_features[i]  = get_match_probabilities(n_episodes*episode_len,
                                        self.volunteers_per_arm,[i for i in range(1,101)],
                                        rf_classifier, rescues_by_user,all_rescue_data,
                                        donation_id_to_latlon, recipient_location_to_latlon, 
                                        user_id_to_latlon)
            self.current_episode_match_probs = self.all_match_probabilities[0]

    def reset_all(self):
        self.instance_count = -1
        self.total_active = 0

        return self.reset_instance()

    def get_average_prob(self,volunteer_num):
        """Get the average match probability across trials
        
        Arguments:
            volunteer_context: Some feature vector, representing volutneers
            Trials: Number of trials to average over
            
        Returns: Average probability, [0,1] float"""

        return np.mean(self.current_episode_match_probs[:,volunteer_num])

    def reset_instance(self):
        """ reset to a new environment instance """
        self.instance_count += 1

        # get new cohort members
        self.cohort_idx       = self.cohort_selection[self.instance_count, :]
        self.agent_idx = []

        if self.contextual:
            self.current_episode_match_probs = self.all_match_probabilities[self.instance_count]

        for idx in self.cohort_idx:
            volunteer_ids = [idx*self.volunteers_per_arm+i for i in range(self.volunteers_per_arm)]
            self.agent_idx += volunteer_ids

        self.features    = self.all_features[self.cohort_idx]
        self.transitions = self.all_transitions[self.cohort_idx] # shape: cohort_size x n_states x 2 x n_states
        self.episode_count = 0

        # current state initialization
        self.timestep    = 0
        self.states      = self.first_init_states[self.instance_count, self.episode_count, :]  # np.copy??
        return self.observe()

    def reset(self):
        self.timestep      = 0
        self.episode_count += 1
        self.states        = self.first_init_states[self.instance_count, self.episode_count, :]
        print(f'instance {self.instance_count}, ep {self.episode_count}')

        return self.observe()

    def fresh_reset(self):
        '''
        This function resets the environment to start over the interaction with arms. The main purpose of
        this function is to sample a new set of arms (with number_arms arms) from the entire population.
        This corresponds to an episode of the restless multi-armed bandit setting but with different
        setup during each episode.

        This simulator also supports infinite time horizon by setting the episode_len to infinity.
        '''

        # Sampling
        sampled_arms     = np.random.choice(a=self.all_population, size=self.cohort_size, replace=False)
        self.features    = self.all_features[sampled_arms]
        self.transitions = self.all_transitions[sampled_arms] # shape: cohort_size x n_states x 2 x n_states

        # Current state initialization
        self.timestep    = 0
        self.states      = self.sample_initial_states(self.cohort_size)

        return self.observe()

    def sample_initial_states(self, cohort_size, prob=0.5):
        '''
        Sampling initial states at random.
        Input:
            cohort_size: the number of arms to be initialized
            prob: the probability of sampling 0 (not engaging state)
        '''
        states = np.random.choice(a=self.number_states, size=cohort_size, p=[prob, 1-prob])
        return states

    def is_terminal(self):
        if self.timestep >= self.episode_len:
            return True
        else:
            return False

    def get_features(self):
        return self.features

    def observe(self):
        return self.states

    def step(self, action):
        assert len(action) == self.cohort_size*self.volunteers_per_arm
        assert np.sum(action) <= self.budget

        reward = self.get_reward(action)

        next_states = np.zeros(self.cohort_size*self.volunteers_per_arm)
        for i in range(self.cohort_size):
            for j in range(self.volunteers_per_arm):
                idx = i*self.volunteers_per_arm + j
                prob = self.transitions[i, self.states[idx], action[idx], :]
                next_state = np.random.choice(a=self.number_states, p=prob)
                next_states[idx] = next_state

        self.states = next_states.astype(int)
        self.timestep += 1

        done = self.is_terminal()

        # print(f'  action {action}, sum {action.sum()}, reward {reward}')

        return self.observe(), reward, done, {}

    def get_reward(self,action=None):
        if self.reward_style == 'state':
            return np.sum(self.states)
        elif self.reward_style == 'match':
            if action is None:
                return 0
            else:
                self.total_active += np.sum(self.states)
                if self.contextual:
                    prod_state = 1-self.states*action*np.array(self.current_episode_match_probs[self.timestep + self.episode_count*self.episode_len])[self.agent_idx]
                else:
                    prod_state = 1-self.states*action*np.array(self.match_probability_list)[self.agent_idx]
                prob_all_inactive = np.prod(prod_state)
                return 1-prob_all_inactive 
        elif self.reward_style == 'submodular': 
            if action is None:
                return 0
            else:
                probs = self.states*action*np.array(self.match_probability_list)[self.agent_idx]
                return np.max(probs) 

        elif self.reward_style == 'custom': 
            if action is None:
                return 0
            else:
                return custom_reward(self.states,action,np.array(self.match_probability_list)[self.agent_idx],self.reward_type,self.reward_parameters)

def random_transition(all_population, n_states, n_actions):
    all_transitions = np.random.random((all_population, n_states, n_actions, n_states))
    all_transitions = all_transitions / np.sum(all_transitions, axis=-1, keepdims=True)
    return all_transitions

def generate_random_context(ndim):
    """Generate a random vector on the surface
        of an ndim unit sphere
        
    Arguments:
        ndim: How many dimensions vector should have

    Returns: Numpy vector of size ndim"""
    vec = np.random.randn(ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def assert_valid_transition(transitions):
    """ check that acting is always good, and starting in good state is always good """

    bad = False
    N, n_states, n_actions, _ = transitions.shape
    for i in range(N):
        for s in range(n_states):
            # ensure acting is always good
            if transitions[i,s,1,1] < transitions[i,s,0,1]:
                bad = True
                print(f'acting should always be good! {i,s} {transitions[i,s,1,1]:.3f} < {transitions[i,s,0,1]:.3f}')

            # assert transitions[i,s,1,1] >= transitions[i,s,0,1] + 1e-6, f'acting should always be good! {transitions[i,s,1,1]:.3f} < {transitions[i,s,0,1]:.3f}'

    for i in range(N):
        for a in range(n_actions):
            # ensure start state is always good
            # assert transitions[i,1,a,1] >= transitions[i,0,a,1] + 1e-6, f'good start state should always be good! {transitions[i,1,a,1]:.3f} < {transitions[i,0,a,1]:.3f}'
            if transitions[i,1,a,1] < transitions[i,0,a,1]:
                bad = True
                print(f'good start state should always be good! {transitions[i,1,a,1]:.3f} < {transitions[i,0,a,1]:.3f}')
    # assert bad != True


def random_valid_transition(all_population, n_states, n_actions):
    """ set initial transition probabilities
    returns array (N, S, A) that shows probability of transitioning to a **good** state

    enforce "valid" transitions: acting is always good, and starting in good state is always good """

    assert n_actions == 2

    transitions = np.random.random((all_population, n_states, n_actions))

    for i in range(all_population):
        for s in range(n_states):
            # ensure acting is always good
            if transitions[i,s,1] < transitions[i,s,0]:
                diff = 1 - transitions[i,s,0]
                transitions[i,s,1] = transitions[i,s,0] + (np.random.rand() * diff)

    for i in range(all_population):
        for a in range(n_actions):
            # ensure starting in good state is always good
            if transitions[i,1,a] < transitions[i,0,a]:
                diff = 1 - transitions[i,0,a]
                transitions[i,1,a] = transitions[i,0,a] + (np.random.rand() * diff)

    full_transitions = np.zeros((all_population, n_states, n_actions, n_states))
    full_transitions[:,:,:,1] = transitions
    full_transitions[:,:,:,0] = 1 - transitions

    # return transitions
    return full_transitions



def random_valid_transition_round_down(all_population, n_states, n_actions):
    """ set initial transition probabilities
    returns array (N, S, A) that shows probability of transitioning to a **good** state

    enforce "valid" transitions: acting is always good, and starting in good state is always good """

    assert n_actions == 2

    transitions = np.random.random((all_population, n_states, n_actions))

    for i in range(all_population):
        for s in range(n_states):
            # ensure acting is always good
            if transitions[i,s,1] < transitions[i,s,0]:
                transitions[i,s,0] = transitions[i,s,1] * np.random.rand()

    for i in range(all_population):
        for a in range(n_actions):
            # ensure starting in good state is always good
            if transitions[i,1,a] < transitions[i,0,a]:
                transitions[i,0,a] = transitions[i,1,a] * np.random.rand()

    full_transitions = np.zeros((all_population, n_states, n_actions, n_states))
    full_transitions[:,:,:,1] = transitions
    full_transitions[:,:,:,0] = 1 - transitions

    return full_transitions


def synthetic_transition_small_window(all_population, n_states, n_actions, low, high):
    """ set initial transition probabilities
    returns array (N, S, A) that shows probability of transitioning to a **good** state

    enforce "valid" transitions: acting is always good, and starting in good state is always good """

    assert n_actions == 2
    assert low < high
    assert 0 < low < 1
    assert 0 < high < 1

    transitions = np.random.random((all_population, n_states, n_actions))

    for i in range(all_population):
        for s in range(n_states):
            # ensure acting is always good
            if transitions[i,s,1] < transitions[i,s,0]:
                transitions[i,s,0] = transitions[i,s,1] * np.random.rand()

    for i in range(all_population):
        for a in range(n_actions):
            # ensure starting in good state is always good
            if transitions[i,1,a] < transitions[i,0,a]:
                transitions[i,0,a] = transitions[i,1,a] * np.random.rand()

    # scale down to a small window .4 to .6
    max_val = np.max(transitions)
    min_val = np.min(transitions)

    transitions = transitions - min_val
    transitions = transitions * (high - low) * (max_val - min_val) + low

    full_transitions = np.zeros((all_population, n_states, n_actions, n_states))
    full_transitions[:,:,:,1] = transitions
    full_transitions[:,:,:,0] = 1 - transitions

    return full_transitions




'''
Testing the functionality of the simulator
'''
if __name__ == '__main__':
    all_population  = 10000
    all_features    = np.arange(all_population)
    all_transitions = random_transition(all_population)
    cohort_size     = 200
    episode_len     = 100
    budget          = 20
    number_states   = 2

    simulator = RMABSimulator(all_population, all_features, all_transitions, cohort_size, episode_len, budget)

    for count in range(10):
        simulator.reset()
        features = simulator.get_features()

        total_reward = 0

        for t in range(episode_len):
            action = np.zeros(cohort_size)
            selection_idx = np.random.choice(a=cohort_size, size=budget, replace=False)
            action[selection_idx] = 1
            action = action.astype(int)

            states, reward, done, _ = simulator.step(action)
            total_reward += reward

        print('total reward: {}'.format(total_reward))


def run_heterogenous_policy(env, n_episodes, n_epochs,discount,policy,seed,per_epoch_function=None,lamb=0,get_memory=False,should_train=False,test_T=0):
    """Wrapper to run policies without needing to go through boilerplate code
    
    Arguments:
        env: Simualtor environment
        n_episodes: How many episodes to run for each epoch
            T = n_episodes * episode_len
        n_epochs: Number of different epochs/cohorts to run
        discount: Float, how much to discount rewards by
        policy: Function that takes in environment, state, budget, and lambda
            produces action as a result
        seed: Random seed for run
        lamb: Float, tradeoff between matching, activity
    
    Returns: Two things
        matching reward - Numpy array of Epochs x T, with rewards for each combo
        activity rate - Average rate of engagement across all volunteers
        We aim to maximize matching reward + lamb*n_arms*activity_rate"""

    N         = env.cohort_size*env.volunteers_per_arm
    n_states  = env.number_states
    n_actions = env.all_transitions.shape[2]
    budget    = env.budget
    T         = env.episode_len * n_episodes
    env.lamb = lamb 

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env.reset_all()

    if should_train:
        all_reward = np.zeros((n_epochs,test_T))
        all_active_rate = np.zeros((n_epochs,test_T))
    else:
        all_reward = np.zeros((n_epochs, T))
        all_active_rate = np.zeros((n_epochs,T))
    env.train_epochs = T-test_T
    env.test_epochs = test_T

    inference_time_taken = 0
    train_time_taken = 0

    for epoch in range(n_epochs):
        if not should_train:
            np.random.seed(seed)
            start = time.time()

        train_time_start = time.time() 

        if epoch != 0: env.reset_instance()
        first_state = env.observe()
        if len(first_state)>20:
            first_state = first_state[:20]

        if per_epoch_function:
            per_epoch_results = per_epoch_function(env,lamb)
        else:
            per_epoch_results = None 

        memory = None 
        for t in range(0, T):
            state = env.observe()
            if should_train:
                if t>=T-test_T:
                    all_active_rate[epoch,t-(T-test_T)] = np.sum(state)/len(state)
            else:
                all_active_rate[epoch,t] = np.sum(state)/len(state)

            action,memory = policy(env,state,budget,lamb,memory,per_epoch_results)
            next_state, reward, done, _ = env.step(action)

            if done and t+1 < T: env.reset()

            if t == T-test_T:
                train_time_taken += time.time()-train_time_start

            if should_train:
                if t == T-test_T:
                    np.random.seed(seed)
                    start = time.time()

                if t < (T-test_T):
                    env.total_active = 0
                else:
                    all_reward[epoch, t-(T-test_T)] = reward
            else:
                all_reward[epoch, t] = reward
        inference_time_taken += time.time()-start 
    env.time_taken = inference_time_taken
    env.train_time = train_time_taken

    print("Took {} time for inference and {} time for training".format(inference_time_taken,train_time_taken))

    if get_memory:
        return all_reward, all_active_rate, memory
    return all_reward, all_active_rate

def get_discounted_reward(global_reward,active_rate,discount,lamb):
    """Compute the discounted combination of global reward and active rate
    
    Arguments: 
        global_reward: numpy array of size n_epochs x T
        active_rate: numpy array of size n_epochs x T
        discount: float, gamma

    Returns: Float, average discounted reward across all epochs"""

    all_rewards = []
    combined_reward = global_reward*(1-lamb) + lamb*active_rate
    num_steps = 5

    step_size = len(global_reward[0])//num_steps

    for epoch in range(len(global_reward)):
        for i in range(num_steps):
            reward = 0
            for t in range(i*step_size,(i+1)*step_size):
                reward += combined_reward[epoch,t]*discount**(t-i*step_size)
            all_rewards.append(reward)
    return np.mean(all_rewards)
