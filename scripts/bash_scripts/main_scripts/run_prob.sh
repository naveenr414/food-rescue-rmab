#!/bin/bash 

cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}
    for n_arms in 4 10 
    do 
        budget_frac=0.5 
        budget=$(echo "${n_arms}*${budget_frac}" | bc)
        budget=$(printf "%.0f" $budget)
        for arm_set_low in 0
        do 
            for arm_set_high in 1
            do 
                if awk -v low="$arm_set_low" -v high="$arm_set_high" 'BEGIN { exit !(high > low) }'; then
                    python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type probability --arm_set_low ${arm_set_low} --arm_set_high ${arm_set_high} --out_folder reward_variation/prob_reward
                    python baselines.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type probability --arm_set_low ${arm_set_low} --arm_set_high ${arm_set_high} --out_folder baselines/all
                fi 
            done 
        done 
    done 
done 