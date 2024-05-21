# Restless Bandits with Global Rewards
This code evaluates and tests restless bandits with a global reward. We develop policies which extend Whittle indices to account for global non-separable rewards. 

The code contains experiments using a synthetic dataset, and real-world data using a Food Rescue dataset is hidden due to confidentiality.

## Directory Structure
The `rmab` folder contains most of the code for core algorithms and functions. 
Whittle-index policies are in `whittle_policies.py`, baselines are in `baseline_policies.py`, MCTS in `mcts_policies.py` and RL policies in `dqn_policies.py`. Bandit simulator code is in `simulator.py`, while food rescue code is in `fr_dynamics.py`. 

The scripts folder uses these functions to run experiments. 
Each experiment is captured by an interactive Jupyter notebook, which is in the `scripts/notebooks` folder, and runs the experiment for one parameter combination. 
Each experiment is also paired with a Python file, which is a converted version of the Jupyter script.
The `scripts/bash_scripts` folder runs different main and ablation scripts by calling the `scripts/notebooks` folder. 
Results are written to the `result` folder. 

## Running Experiments
First, create all the folders needed to run experiments by running
```bash
bash scripts/bash_scripts/main_scripts/create_folders.sh
```
Then run 
```bash
pip install .
```

Experiments are run from the `scripts/notebooks` folder. For example, to run all experiments with the Linear reward function, from within the `scripts/notebooks` folder, run
```
bash ../main_scripts/run_linear.sh
```
The `Plotting.ipynb` notebook uses the information from the results folder to create plots. 

## Requirements
All requirements are contained in the ``environment.yaml'' file. 
To creaet an anaconda environment from this, run
```
conda env create --file environment.yaml
```

## Running custom policies
To run custom policies, define a function that takes in an environment and a state, then returns an action
For example, to define the random policy: 
```
def random_policy(env,state,budget,lamb,memory, per_epoch_results):
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
    selected_idx = random.sample(list(range(N)), budget)
    action = np.zeros(N, dtype=np.int8)
    action[selected_idx] = 1

    return action, None
```