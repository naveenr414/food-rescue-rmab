#!/bin/bash

# Main Scripts
bash ../bash_scripts/main_scripts/run_max.sh 
bash ../bash_scripts/main_scripts/run_linear.sh 
bash ../bash_scripts/main_scripts/run_set.sh 
bash ../bash_scripts/main_scripts/run_prob.sh 
bash ../bash_scripts/main_scripts/vary_n_k.sh 
bash ../bash_scripts/main_scripts/run_food_rescue.sh

# # Ablation Scripts
bash ../bash_scripts/ablation_scripts/run_pure_mcts.sh
bash ../bash_scripts/ablation_scripts/run_pure_rl.sh
bash ../bash_scripts/ablation_scripts/run_synthetic_transitions.sh