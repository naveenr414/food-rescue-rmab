# Restless Bandits with Matching
This code evaluates and tests restless bandits in a matching context, including both the activity of arms, and their match success. 

The code contains experiments using a synthetic dataset, and real-world data using a Food Rescue dataset is hidden due to confidentiality.

## Directory Structure
The `rmab` folder contains most of the code for core algorithms and functions. Whittle-index related code is in `compute_whittle.py`, `uc_whittle.py`, and `ucw_value.py`. Food-rescue related code is in `fr_dynamics.py` and `database.py.` Bandit simulator code is in `simulator.py`, while matching algorithms are in `uc_whittle.py` (TODO: Move this somewhere else). 

The scripts folder uses these functions to run experiments. Each experiment is captured by an interactive Jupyter notebook, which runs the experiment for one parameter combination, and a Python file, which allows for many different run combinations. Bash files run these python files across many configurations, and the results are stored in the results folder. 

## Running Experiments
Experiments are run from the scripts folder. For example, to compare activity for various bandit algorithms, from within the scripts folder, run
```
bash run_better_bandit.sh
```
To compare UC Whittle vs. Normal-distribution performance for an indvidual configuration, run 
```
python better_bandit.py --seed 42 --dataset synthetic --n_arms 8 --save_name normal --use_date
```
Doing so will compare bandit algorithms for 8 arms, saving it as `results/better_bandit/normal_42_{date}.json`` 

To do this for matching algorithms and compare match rate, run 
```
bash run_matching_bandit.sh
```
Similarly, to compute pareto frontiers, run
```
bash run_combined_bandit.sh
```
The `Plotting.ipynb` notebook uses the information from the results folder to create plots. 

## Requirements
All requirements are contained in the ``environment.yaml'' file. 
To creaet an anaconda environment from this, run
```
conda env create --file environment.yaml
```
