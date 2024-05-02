#!/bin/bash

# Main Scripts
bash ../bash_scripts/main_scripts/vary_n_k.sh 
bash ../bash_scripts/main_scripts/run_max.sh 
bash ../bash_scripts/main_scripts/run_set.sh 
bash ../bash_scripts/main_scripts/run_prob.sh 
bash ../bash_scripts/main_scripts/run_food_rescue.sh
bash ../bash_scripts/main_scripts/run_baselines.sh

# Ablation Scripts
bash ../bash_scripts/ablation_scripts/run_pure_mcts.sh
