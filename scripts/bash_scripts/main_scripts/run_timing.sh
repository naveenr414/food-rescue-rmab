#!/bin/bash 

cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}

    for volunteers in 5 10 15 20 25 
    do 
        echo ${volunteers}
        budget_frac=0.5
        budget=$(echo "${volunteers}*${budget_frac}" | bc)
        budget=$(printf "%.0f" $budget)
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type set_cover --arm_set_low 6 --arm_set_high 8 --universe_size 20 --run_rate_limits --out_folder reward_variation/subset_reward
    done 
done 
