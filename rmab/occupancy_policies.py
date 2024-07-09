""" Occupancy-Based Algorithms for matching, activity """

import numpy as np
from rmab.utils import one_hot, get_average_context, custom_reward_contextual, Memoizer, custom_reward, avg_reward_contextual
from rmab.whittle_policies import shapley_index_custom, whittle_index, compute_regression
import random 
import scipy 

import gurobipy as gp
from gurobipy import GRB
from sklearn.linear_model import LinearRegression, Ridge

def occupancy_measure_shapley(env,state,budget,lamb,memory, per_epoch_results):
    """Random policy that randomly notifies budget arms
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    N = len(state)

    if memory == None:
        memory_whittle = Memoizer('optimal')
        memory_shapley = np.array(shapley_index_custom(env,np.ones(len(state)),{})[0])
        for i in range(2):
            s = [i for _ in range(len(state))]
            whittle_index(env,s,budget,lamb,memory_whittle,reward_function="combined",match_probs=memory_shapley)
        chi = None 
        r = None     
    else:
        memory_whittle, memory_shapley, chi, r = memory 

    if env.timestep == 0:
        T = env.episode_len
        N = len(state)
        
        r = np.zeros((N,2,2))

        for i in range(N):
            r[i,1,:] = lamb/N 
            r[i,1,1] += (1-lamb)*memory_shapley[i]

        model = gp.Model("LP")
        model.setParam('OutputFlag', 0)
        mu = model.addVars(T, N, 2, 2, name="mu", vtype=GRB.CONTINUOUS, lb=0,ub=1)
        model.setObjective(gp.quicksum(mu[t, i, s, a] * r[i, s, a] for t in range(T) for i in range(N) for s in range(2) for a in range(2)), GRB.MAXIMIZE)
        for t in range(T):
            model.addConstr(gp.quicksum(a * mu[t, i, s, a] for i in range(N) for s in range(2) for a in range(2)) <= budget, name=f"budget_t{t}")

        for t in range(1, T):
            for i in range(N):
                for s in range(2):
                    model.addConstr(gp.quicksum(mu[t, i, s, a] for a in range(2)) == gp.quicksum(mu[t-1, i, s_prime, a_prime] * env.transitions[i, s_prime, a_prime,s] for s_prime in range(2) for a_prime in range(2)), name=f"flow_i{i}_t{t}")

        for i in range(N):
            for s in range(2):
                actual_state_value = state[i] 
                model.addConstr(gp.quicksum(mu[0,i,s,a] for a in range(2)) == int(actual_state_value == s))

        model.optimize()
        solution = model.getAttr('x', mu)

        chi = np.zeros((N,T,2,2))

        for i in range(N):
            for t in range(T):
                for s in range(2):
                    for a in range(2):
                        if sum([solution[t,i,s,a_prime] for a_prime in range(2)]) != 0:
                            chi[i,t,s,a] = solution[t,i,s,a]/sum([solution[t,i,s,a_prime] for a_prime in range(2)])
        

    phi = np.zeros((N))
    epsilon = 0.0001

    for i in range(N):
        phi[i] = chi[i,env.timestep,state[i],1] * (r[i,state[i],1]+epsilon)

    selected_idx = np.argsort(phi)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, (memory_whittle, memory_shapley, chi, r)

def occupancy_measure_shapley_contextual(env,state,budget,lamb,memory, per_epoch_results):
    """Random policy that randomly notifies budget arms
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    N = len(state)

    if memory == None:
        chi = None 
        r = None     
    else:
        chi, r = memory 

    explore_episodes = 1

    if env.episode_count < explore_episodes:
        N = len(state)
        selected_idx = random.sample(list(range(N)), budget)
        action = np.zeros(N, dtype=np.int8)
        action[selected_idx] = 1

        return action, (chi,r)

    T = env.episode_len-env.timestep
    N = len(state)

    r = np.zeros((T,N,2,2))

    if env.episode_count >= explore_episodes and env.episode_count>0:
        current_timestep = env.episode_len*env.episode_count + env.timestep
        memory_shapley = np.array(compute_regression(env))
        current_reward = [custom_reward_contextual(one_hot(i,len(state)),one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters,env.context) for i in range(len(state))]

        for t in range(T):
            if t == 0:
                for i in range(N):
                    r[t,i,1,:] = lamb/N 
                    r[t,i,1,1] += (1-lamb)*current_reward[i]            
            else:
                for i in range(N):
                    r[t,i,1,:] = lamb/N 
                    r[t,i,1,1] += (1-lamb)*memory_shapley[i]


        model = gp.Model("LP")
        model.setParam('OutputFlag', 0)
        mu = model.addVars(T, N, 2, 2, name="mu", vtype=GRB.CONTINUOUS, lb=0,ub=1)
        model.setObjective(gp.quicksum(mu[t, i, s, a] * r[t,i, s, a] for t in range(T) for i in range(N) for s in range(2) for a in range(2)), GRB.MAXIMIZE)
        for t in range(T):
            model.addConstr(gp.quicksum(a * mu[t, i, s, a] for i in range(N) for s in range(2) for a in range(2)) <= budget, name=f"budget_t{t}")

        for t in range(1, T):
            for i in range(N):
                for s in range(2):
                    model.addConstr(gp.quicksum(mu[t, i, s, a] for a in range(2)) == gp.quicksum(mu[t-1, i, s_prime, a_prime] * env.transitions[i, s_prime, a_prime,s] for s_prime in range(2) for a_prime in range(2)), name=f"flow_i{i}_t{t}")

        for i in range(N):
            for s in range(2):
                actual_state_value = state[i] 
                model.addConstr(gp.quicksum(mu[0,i,s,a] for a in range(2)) == int(actual_state_value == s))

        model.optimize()
        solution = model.getAttr('x', mu)

        chi = np.zeros((N,T,2,2))

        for i in range(N):
            for t in range(T):
                for s in range(2):
                    for a in range(2):
                        if sum([solution[t,i,s,a_prime] for a_prime in range(2)]) != 0:
                            chi[i,t,s,a] = solution[t,i,s,a]/sum([solution[t,i,s,a_prime] for a_prime in range(2)])
    


    phi = np.zeros((N))
    epsilon = 0.0001

    for i in range(N):
        phi[i] = chi[i,0,state[i],1] * (r[0,i,state[i],1]+epsilon)

    selected_idx = np.argsort(phi)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, (chi, r)


def occupancy_measure_linear(env,state,budget,lamb,memory, per_epoch_results):
    """Random policy that randomly notifies budget arms
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    N = len(state)

    if memory == None:
        if env.use_context:
            match_probs = [avg_reward_contextual(one_hot(i,len(state)),one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters) for i in range(len(state))]
        else:
            match_probs = [custom_reward(one_hot(i,len(state)),one_hot(i,len(state)),np.array(env.match_probability_list)[env.agent_idx],env.reward_type,env.reward_parameters) for i in range(len(state))]
        chi = None 
        r = None   
    else:
        match_probs, chi, r = memory 

    if env.timestep == 0:
        T = env.episode_len
        N = len(state)
        
        r = np.zeros((N,2,2))

        for i in range(N):
            r[i,1,:] = lamb/N 
            r[i,1,1] += (1-lamb)*match_probs[i]

        model = gp.Model("LP")
        model.setParam('OutputFlag', 0)
        mu = model.addVars(T, N, 2, 2, name="mu", vtype=GRB.CONTINUOUS, lb=0,ub=1)
        model.setObjective(gp.quicksum(mu[t, i, s, a] * r[i, s, a] for t in range(T) for i in range(N) for s in range(2) for a in range(2)), GRB.MAXIMIZE)
        for t in range(T):
            model.addConstr(gp.quicksum(a * mu[t, i, s, a] for i in range(N) for s in range(2) for a in range(2)) <= budget, name=f"budget_t{t}")

        for t in range(1, T):
            for i in range(N):
                for s in range(2):
                    model.addConstr(gp.quicksum(mu[t, i, s, a] for a in range(2)) == gp.quicksum(mu[t-1, i, s_prime, a_prime] * env.transitions[i, s_prime, a_prime,s] for s_prime in range(2) for a_prime in range(2)), name=f"flow_i{i}_t{t}")

        for i in range(N):
            for s in range(2):
                actual_state_value = state[i] 
                model.addConstr(gp.quicksum(mu[0,i,s,a] for a in range(2)) == int(actual_state_value == s))

        model.optimize()
        solution = model.getAttr('x', mu)

        chi = np.zeros((N,T,2,2))

        for i in range(N):
            for t in range(T):
                for s in range(2):
                    for a in range(2):
                        if sum([solution[t,i,s,a_prime] for a_prime in range(2)]) != 0:
                            chi[i,t,s,a] = solution[t,i,s,a]/sum([solution[t,i,s,a_prime] for a_prime in range(2)])
        

    phi = np.zeros((N))
    epsilon = 0.0001

    for i in range(N):
        phi[i] = chi[i,env.timestep,state[i],1] * (r[i,state[i],1]+epsilon)

    selected_idx = np.argsort(phi)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, (match_probs, chi, r)

def occupancy_measure_learn_shapley(env,state,budget,lamb,memory, per_epoch_results):
    """Random policy that randomly notifies budget arms
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    N = len(state)

    if memory == None:
        chi = None 
        r = None     
    else:
        chi, r = memory 

    explore_episodes = 1

    if env.episode_count < explore_episodes:
        N = len(state)
        selected_idx = random.sample(list(range(N)), budget)
        action = np.zeros(N, dtype=np.int8)
        action[selected_idx] = 1

        return action, (chi,r)

    if env.timestep == 0:
        T = env.episode_len
        N = len(state)

        r = np.zeros((N,2,2))

        if env.episode_count >= explore_episodes and env.episode_count>0:
            current_timestep = env.episode_len*env.episode_count + env.timestep
            seen_rewards = env.all_reward[0,:current_timestep]
            past_states = np.array(env.past_states)[:current_timestep]
            past_actions = np.array(env.past_actions)[:current_timestep]
            model = gp.Model("LP")
            model.setParam('OutputFlag', 0)

            # Reward Shapley
            ms = model.addVars(N, name="ms", vtype=GRB.CONTINUOUS, lb=0,ub=20)
            model.setObjective(gp.quicksum((gp.quicksum(ms[i]*past_actions[t,i]*past_states[t,i] for i in range(N))-seen_rewards[t])**2 for t in range(current_timestep)), GRB.MINIMIZE)
            # model.setObjective(gp.quicksum(ms[i] for i in range(N)), GRB.MAXIMIZE)

            for t in range(current_timestep):
                model.addConstr(gp.quicksum(ms[i]*past_actions[t,i]*past_states[t,i] for i in range(N)) <= seen_rewards[t])
            model.optimize()
            memory_shapley = model.getAttr('x', ms)
            memory_shapley = [memory_shapley[i] for i in range(len(memory_shapley))]

            # TODO: Remove this
            X = past_states*past_actions
            Y = seen_rewards

            if "random" not in env.reward_type:
                _, indices = np.unique(X, axis=0, return_index=True)

                # Sort indices to maintain original order
                sorted_indices = np.sort(indices)

                X = X[sorted_indices]
                Y = Y[sorted_indices]

            reg = LinearRegression(fit_intercept=False).fit(X,Y)
            memory_shapley = reg.coef_

            real_memory_shapley = np.array(compute_regression(env))
            env.predicted_regression = memory_shapley
            env.actual_regression = real_memory_shapley

            arm_frequencies = np.sum(past_actions[current_timestep-env.episode_len:current_timestep]*past_states[current_timestep-env.episode_len:current_timestep],axis=0)
            diff = np.abs(memory_shapley-real_memory_shapley)/real_memory_shapley
            print("Mem",memory_shapley)
            print("Real",real_memory_shapley)
            # print("arm_freq",arm_frequencies)
            print(len(X),np.mean(np.abs(memory_shapley[arm_frequencies>0]-real_memory_shapley[arm_frequencies>0])))

            for i in range(N):
                r[i,1,:] = lamb/N 
                r[i,1,1] += (1-lamb)*memory_shapley[i]        
        else:
            for i in range(N):
                r[i,1,:] = lamb/N 
                r[i,1,1] += (1-lamb)*random.random()*20


        model = gp.Model("LP")
        model.setParam('OutputFlag', 0)
        mu = model.addVars(T, N, 2, 2, name="mu", vtype=GRB.CONTINUOUS, lb=0,ub=1)
        model.setObjective(gp.quicksum(mu[t, i, s, a] * r[i, s, a] for t in range(T) for i in range(N) for s in range(2) for a in range(2)), GRB.MAXIMIZE)
        for t in range(T):
            model.addConstr(gp.quicksum(a * mu[t, i, s, a] for i in range(N) for s in range(2) for a in range(2)) <= budget, name=f"budget_t{t}")

        for t in range(1, T):
            for i in range(N):
                for s in range(2):
                    model.addConstr(gp.quicksum(mu[t, i, s, a] for a in range(2)) == gp.quicksum(mu[t-1, i, s_prime, a_prime] * env.transitions[i, s_prime, a_prime,s] for s_prime in range(2) for a_prime in range(2)), name=f"flow_i{i}_t{t}")

        for i in range(N):
            for s in range(2):
                actual_state_value = state[i] 
                model.addConstr(gp.quicksum(mu[0,i,s,a] for a in range(2)) == int(actual_state_value == s))

        model.optimize()
        solution = model.getAttr('x', mu)

        chi = np.zeros((N,T,2,2))

        for i in range(N):
            for t in range(T):
                for s in range(2):
                    for a in range(2):
                        if sum([solution[t,i,s,a_prime] for a_prime in range(2)]) != 0:
                            chi[i,t,s,a] = solution[t,i,s,a]/sum([solution[t,i,s,a_prime] for a_prime in range(2)])
    


    phi = np.zeros((N))
    epsilon = 0.0001

    for i in range(N):
        phi[i] = chi[i,env.timestep,state[i],1] * (r[i,state[i],1]+epsilon)

    selected_idx = np.argsort(phi)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, (chi, r)

def occupancy_measure_learn_shapley_contextual(env,state,budget,lamb,memory, per_epoch_results):
    """Random policy that randomly notifies budget arms
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    N = len(state)

    if memory == None:
        chi = None 
        r = None     
    else:
        chi, r = memory 

    explore_episodes = 1

    if env.episode_count < explore_episodes:
        N = len(state)
        selected_idx = random.sample(list(range(N)), budget)
        action = np.zeros(N, dtype=np.int8)
        action[selected_idx] = 1

        return action, (chi,r)

    T = env.episode_len-env.timestep
    N = len(state)

    r = np.zeros((T,N,2,2))

    if env.episode_count >= explore_episodes and env.episode_count>0:
        current_timestep = env.episode_len*env.episode_count + env.timestep
        seen_rewards = env.all_reward[0,:current_timestep]
        past_states = np.array(env.past_states)[:current_timestep]
        past_actions = np.array(env.past_actions)[:current_timestep]
        past_contexts = np.array(env.past_contexts)[:current_timestep]
        past_contexts = past_contexts[:250]
        lipschitz_constant = 1

        model = gp.Model("LP")
        model.setParam('OutputFlag', 0)

        # Reward Shapley
        # ms = model.addVars(N, name="ms", vtype=GRB.CONTINUOUS, lb=0,ub=20)
        # model.setObjective(gp.quicksum((gp.quicksum(ms[i]*past_actions[t,i]*past_states[t,i] for i in range(N))-seen_rewards[t])**2 for t in range(current_timestep)), GRB.MINIMIZE)
        # # model.setObjective(gp.quicksum(ms[i] for i in range(N)), GRB.MAXIMIZE)

        # for t in range(current_timestep):
        #     model.addConstr(gp.quicksum(ms[i]*past_actions[t,i]*past_states[t,i] for i in range(N)) <= seen_rewards[t])
        # model.optimize()
        # memory_shapley = model.getAttr('x', ms)
        # memory_shapley = [memory_shapley[i] for i in range(len(memory_shapley))]

        # TODO: Remove this
        X = past_states*past_actions
        Y = seen_rewards

        if "random" not in env.reward_type:
            _, indices = np.unique(X, axis=0, return_index=True)

            # Sort indices to maintain original order
            sorted_indices = np.sort(indices)

            X = X[sorted_indices]
            Y = Y[sorted_indices]

        reg = LinearRegression(fit_intercept=False).fit(X,Y)
        memory_shapley = reg.coef_

        real_memory_shapley = np.array(compute_regression(env))
        env.predicted_regression = memory_shapley
        env.actual_regression = real_memory_shapley

        arm_frequencies = np.sum(past_actions[current_timestep-env.episode_len:current_timestep]*past_states[current_timestep-env.episode_len:current_timestep],axis=0)
        diff = np.abs(memory_shapley-real_memory_shapley)/real_memory_shapley
        # print("Mem",memory_shapley)
        # print("Real",real_memory_shapley)
        # print("arm_freq",arm_frequencies)
        # print(len(X),np.mean(np.abs(memory_shapley[arm_frequencies>0]-real_memory_shapley[arm_frequencies>0])))

        all_ms = []
        for idx in range(N):
            ms = model.addVars(N, name="ms", vtype=GRB.CONTINUOUS, lb=0,ub=20)
            model.setObjective(ms[idx], GRB.MAXIMIZE)

            for i in range(len(past_contexts)):
                diff_context = np.sum(np.abs(past_contexts[i]-env.context))
                max_reward_diff = diff_context
                model.addConstr(gp.quicksum(ms[j]*past_states[i][j]*past_actions[i][j] for j in range(N))<= seen_rewards[i]+max_reward_diff)
                # model.addConstr(gp.quicksum(ms[j]*past_states[i][j]*past_actions[i][j] for j in range(N))>= seen_rewards[i]-max_reward_diff)
            model.optimize()
            all_ms.append(model.getAttr('x',ms)[idx])


        for t in range(T):
            if t == 0:
                for i in range(N):
                    r[t,i,1,:] = lamb/N 
                    r[t,i,1,1] += (1-lamb)*all_ms[i]     
            else:
                for i in range(N):
                    r[t,i,1,:] = lamb/N 
                    r[t,i,1,1] += (1-lamb)*memory_shapley[i]        


        model = gp.Model("LP")
        model.setParam('OutputFlag', 0)
        mu = model.addVars(T, N, 2, 2, name="mu", vtype=GRB.CONTINUOUS, lb=0,ub=1)
        model.setObjective(gp.quicksum(mu[t, i, s, a] * r[t,i, s, a] for t in range(T) for i in range(N) for s in range(2) for a in range(2)), GRB.MAXIMIZE)
        for t in range(T):
            model.addConstr(gp.quicksum(a * mu[t, i, s, a] for i in range(N) for s in range(2) for a in range(2)) <= budget, name=f"budget_t{t}")

        for t in range(1, T):
            for i in range(N):
                for s in range(2):
                    model.addConstr(gp.quicksum(mu[t, i, s, a] for a in range(2)) == gp.quicksum(mu[t-1, i, s_prime, a_prime] * env.transitions[i, s_prime, a_prime,s] for s_prime in range(2) for a_prime in range(2)), name=f"flow_i{i}_t{t}")

        for i in range(N):
            for s in range(2):
                actual_state_value = state[i] 
                model.addConstr(gp.quicksum(mu[0,i,s,a] for a in range(2)) == int(actual_state_value == s))

        model.optimize()
        solution = model.getAttr('x', mu)

        chi = np.zeros((N,T,2,2))

        for i in range(N):
            for t in range(T):
                for s in range(2):
                    for a in range(2):
                        if sum([solution[t,i,s,a_prime] for a_prime in range(2)]) != 0:
                            chi[i,t,s,a] = solution[t,i,s,a]/sum([solution[t,i,s,a_prime] for a_prime in range(2)])
    


    phi = np.zeros((N))
    epsilon = 0.0001

    for i in range(N):
        phi[i] = chi[i,0,state[i],1] * (r[0,i,state[i],1]+epsilon)

    selected_idx = np.argsort(phi)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, (chi, r)


def occupancy_measure_learn_shapley_confidence(env,state,budget,lamb,memory, per_epoch_results):
    """Random policy that randomly notifies budget arms
    
    Arguments:
        env: Simulator environment
        state: Numpy array with 0-1 states for each agent
        budget: Integer, max agents to select
        lamb: Lambda, float, tradeoff between matching vs. activity
        memory: Any information passed from previous epochs; unused here
        per_epoch_results: Any information computed per epoch; unused here
    
    Returns: Actions, numpy array of 0-1 for each agent, and memory=None"""

    N = len(state)

    if memory == None:
        chi = None 
        r = None     
    else:
        chi, r = memory 

    explore_episodes = 1

    if env.episode_count < explore_episodes:
        N = len(state)
        selected_idx = random.sample(list(range(N)), budget)
        action = np.zeros(N, dtype=np.int8)
        action[selected_idx] = 1

        return action, (chi,r)

    if env.timestep == 0:
        T = env.episode_len
        N = len(state)

        r = np.zeros((N,2,2))
        epsilon = 0.01

        def get_beta(val,timestep):
            return epsilon*val

        if env.episode_count >= explore_episodes and env.episode_count>0:
            current_timestep = env.episode_len*env.episode_count + env.timestep
            seen_rewards = env.all_reward[0,:current_timestep]
            past_states = np.array(env.past_states)[:current_timestep]
            past_actions = np.array(env.past_actions)[:current_timestep]
            
            X = past_states*past_actions
            Y = seen_rewards

            if "random" not in env.reward_type:
                _, indices = np.unique(X, axis=0, return_index=True)

                # Sort indices to maintain original order
                sorted_indices = np.sort(indices)

                X = X[sorted_indices]
                Y = Y[sorted_indices]

            reg = LinearRegression(fit_intercept=False).fit(X,Y)
            memory_shapley = reg.coef_
            memory_shapley = np.array(memory_shapley) + get_beta(np.mean(memory_shapley),env.episode_count)

            real_memory_shapley = np.array(compute_regression(env))
            env.predicted_regression = memory_shapley
            env.actual_regression = real_memory_shapley

            arm_frequencies = np.sum(past_actions[current_timestep-env.episode_len:current_timestep]*past_states[current_timestep-env.episode_len:current_timestep],axis=0)
            diff = np.abs(memory_shapley-real_memory_shapley)/real_memory_shapley
            # print("Mem",memory_shapley)
            # print("Real",real_memory_shapley)
            # print("arm_freq",arm_frequencies)
            # print(len(X),np.mean(np.abs(memory_shapley[arm_frequencies>0]-real_memory_shapley[arm_frequencies>0])))

            for i in range(N):
                r[i,1,:] = lamb/N 
                r[i,1,1] += (1-lamb)*memory_shapley[i]        
        else:
            for i in range(N):
                r[i,1,:] = lamb/N 
                r[i,1,1] += (1-lamb)*random.random()*20


        model = gp.Model("LP")
        model.setParam('OutputFlag', 0)
        mu = model.addVars(T, N, 2, 2, name="mu", vtype=GRB.CONTINUOUS, lb=0,ub=1)
        model.setObjective(gp.quicksum(mu[t, i, s, a] * r[i, s, a] for t in range(T) for i in range(N) for s in range(2) for a in range(2)), GRB.MAXIMIZE)
        for t in range(T):
            model.addConstr(gp.quicksum(a * mu[t, i, s, a] for i in range(N) for s in range(2) for a in range(2)) <= budget, name=f"budget_t{t}")

        for t in range(1, T):
            for i in range(N):
                for s in range(2):
                    model.addConstr(gp.quicksum(mu[t, i, s, a] for a in range(2)) == gp.quicksum(mu[t-1, i, s_prime, a_prime] * env.transitions[i, s_prime, a_prime,s] for s_prime in range(2) for a_prime in range(2)), name=f"flow_i{i}_t{t}")

        for i in range(N):
            for s in range(2):
                actual_state_value = state[i] 
                model.addConstr(gp.quicksum(mu[0,i,s,a] for a in range(2)) == int(actual_state_value == s))

        model.optimize()
        solution = model.getAttr('x', mu)

        chi = np.zeros((N,T,2,2))

        for i in range(N):
            for t in range(T):
                for s in range(2):
                    for a in range(2):
                        if sum([solution[t,i,s,a_prime] for a_prime in range(2)]) != 0:
                            chi[i,t,s,a] = solution[t,i,s,a]/sum([solution[t,i,s,a_prime] for a_prime in range(2)])
    


    phi = np.zeros((N))
    epsilon = 0.0001

    for i in range(N):
        phi[i] = chi[i,env.timestep,state[i],1] * (r[i,state[i],1]+epsilon)

    selected_idx = np.argsort(phi)[-budget:][::-1]
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, (chi, r)
