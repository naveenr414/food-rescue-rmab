import gym
import numpy as np
import random 
import gymnasium.spaces as spaces


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
            discount,number_states=2,reward_style='state',match_probability=0.5,match_probability_list = [],TIME_PER_RUN=10.0):
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

        if self.match_probability_list == []:
            self.match_probability_list = [match_probability for i in range(self.cohort_size * self.volunteers_per_arm)]

        assert_valid_transition(all_transitions)

        # set up options for the multiple instances
        # track the first random initial state
        self.instance_count = 0
        self.episode_count  = 0
        self.timestep       = 0
        self.total_active = 0

        # track indices of cohort members
        self.cohort_selection  = np.zeros((n_instances, cohort_size)).astype(int)
        self.first_init_states = np.zeros((n_instances, n_episodes, cohort_size*volunteers_per_arm)).astype(int)
        for i in range(n_instances):
            self.cohort_selection[i, :] = np.random.choice(a=self.all_population, size=self.cohort_size, replace=False)
            print('cohort', self.cohort_selection[i, :])
            for ep in range(n_episodes):
                self.first_init_states[i, ep, :] = self.sample_initial_states(self.cohort_size*self.volunteers_per_arm)

    def reset_all(self):
        self.instance_count = -1
        self.total_active = 0

        return self.reset_instance()

    def reset_instance(self):
        """ reset to a new environment instance """
        self.instance_count += 1

        # get new cohort members
        self.cohort_idx       = self.cohort_selection[self.instance_count, :]
        self.agent_idx = []

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
                # TODO: Make sure this is correct
                prod_state = 1-self.states*action*np.array(self.match_probability_list)[self.agent_idx]

                prob_all_inactive = np.prod(prod_state)


                # for i in range(len(self.states)):
                #     prob_all_inactive *= (1-self.states[i]*action[i]*np.array(self.match_probability_list)[self.agent_idx][i])                
                return 1-prob_all_inactive 

class RMABSimulatorOpenRL(gym.Env):
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

    def __init__(self, all_population, all_features, all_transitions, cohort_size, episode_len, n_instances, n_episodes, budget,
            number_states=2,reward_style='state',match_probability=0.5,lamb=0,**kwargs):
        '''
        Initialization
        '''
        self.all_population  = all_population
        self.all_features    = all_features
        self.all_transitions = all_transitions
        self.cohort_size     = cohort_size
        self.action_space = spaces.MultiBinary(self.cohort_size)  # 0-1 vectors for each agent
        self.observation_space = spaces.Box(0, 1, [self.cohort_size], dtype=np.float32)  # Example observation space
        self.budget          = budget
        self.parallel_env_num = 1
        self.env_name = "rmab_simulator"
        self.number_states   = number_states
        self.episode_len     = episode_len
        self.n_episodes      = n_episodes   # total number of episodes per epoch
        self.n_instances     = n_instances  # n_epochs: number of separate transitions / instances
        self.reward_style = reward_style # Should we get reward style based on states or matches
        self.match_probability = match_probability
        self.agent_num = cohort_size
        self.lamb = lamb 

        assert_valid_transition(all_transitions)

        # set up options for the multiple instances
        # track the first random initial state
        self.instance_count = 0
        self.episode_count  = 0
        self.timestep       = 0
        self.total_active = 0
        self.dim = 1

        # track indices of cohort members
        self.cohort_selection  = np.zeros((n_instances, cohort_size)).astype(int)
        self.first_init_states = np.zeros((n_instances, n_episodes, cohort_size)).astype(int)
        for i in range(n_instances):
            self.cohort_selection[i, :] = np.random.choice(a=self.all_population, size=self.cohort_size, replace=False)
            print('cohort', self.cohort_selection[i, :])
            for ep in range(n_episodes):
                self.first_init_states[i, ep, :] = self.sample_initial_states(self.cohort_size)

        self.cohort_idx       = self.cohort_selection[0, :]
        self.transitions = self.all_transitions[self.cohort_idx] # shape: cohort_size x n_states x 2 x n_states
        

    def reset_all(self):
        self.instance_count = -1
        self.total_active = 0
        # self.reset_type = 'normal'

        return self.reset_instance()

    def reset_instance(self):
        """ reset to a new environment instance """
        self.instance_count += 1

        # get new cohort members
        self.cohort_idx       = self.cohort_selection[self.instance_count, :]
        self.features    = self.all_features[self.cohort_idx]
        self.transitions = self.all_transitions[self.cohort_idx] # shape: cohort_size x n_states x 2 x n_states
        self.episode_count = 0

        # current state initialization
        self.timestep    = 0
        self.states      = self.first_init_states[self.instance_count, self.episode_count, :]  # np.copy??


        return self.observe(), {}

    def reset(self,seed=42,**kwargs):
        if kwargs['options'] != None:
            reset_type = kwargs['options']['reset_type']
        else:
            reset_type = 'normal'
        if reset_type == 'to_0':
            return self.reset_to_episode_0()
        elif reset_type == 'full':
            return self.reset_all()
        elif reset_type == 'instance':
            return self.reset_instance()
        else:
            self.timestep      = 0
            self.episode_count += 1
            self.episode_count %= self.first_init_states.shape[1]

            self.states        = self.first_init_states[self.instance_count, self.episode_count, :]
            return self.observe(), {}

    def reset_to_episode_0(self):
        # self.reset_type = 'normal'
        self.timestep = 0
        self.episode_count = 0
        self.total_active = 0
        self.states        = self.first_init_states[self.instance_count, self.episode_count, :]
        return self.observe(), {}


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
        assert len(action) == self.cohort_size

        reward = self.get_reward(action)

        next_states = np.zeros(self.cohort_size)
        for i in range(self.cohort_size):
            prob = self.transitions[i, self.states[i], action[i], :]
            next_state = np.random.choice(a=self.number_states, p=prob)
            next_states[i] = next_state

        self.states = next_states.astype(int)
        self.timestep += 1

        done = self.is_terminal()

        return self.observe(), reward, done, {'total_active':self.total_active}

    def get_reward(self,action=None):
        if self.reward_style == 'state':
            return np.sum(self.states)
        elif self.reward_style == 'match':
            if action is None:
                return 0
            if np.sum(action)>self.budget:
                return -1000
            num_active = np.sum(self.states*action)
            self.total_active += np.sum(self.states)

            prob_all_inactive = (1-self.match_probability)**num_active 
            return 1-prob_all_inactive
        elif self.reward_style == 'combined':
            if action is None:
                return 0
            if np.sum(action)>self.budget:
                return 0
            num_active = np.sum(self.states*action)
            self.total_active += np.sum(self.states)

            prob_all_inactive = (1-self.match_probability)**num_active 
            return (1-prob_all_inactive) + self.lamb * np.sum(self.states)



def random_transition(all_population, n_states, n_actions):
    all_transitions = np.random.random((all_population, n_states, n_actions, n_states))
    all_transitions = all_transitions / np.sum(all_transitions, axis=-1, keepdims=True)
    return all_transitions


def assert_valid_transition(transitions):
    """ check that acting is always good, and starting in good state is always good """

    bad = False
    N, n_states, n_actions, _ = transitions.shape
    for i in range(N):
        for s in range(n_states):
            # ensure acting is always good
            if transitions[i,s,1,1] < transitions[i,s,0,1]:
                bad = True
                print(f'acting should always be good! {transitions[i,s,1,1]:.3f} < {transitions[i,s,0,1]:.3f}')

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