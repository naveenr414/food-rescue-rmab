#!/bin/bash 

cd scripts/notebooks

for seed in $(seq 43 43); 
do 
    echo ${seed}

    for n_arms in 4 10 
    do 
        budget_frac=0.5 
        budget=$(echo "${n_arms}*${budget_frac}" | bc)
        budget=$(printf "%.0f" $budget)
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type linear --arm_set_low 0 --arm_set_high 1 --out_folder reward_variation/linear_reward
        python baselines.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type linear --arm_set_low 0 --arm_set_high 1 --out_folder baselines/all
    done 
done  
