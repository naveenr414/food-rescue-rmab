#!/bin/bash 

cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}

    for episode_len in 5000 50000
    do 
        for n_arms in 4 10
        do 
            budget_frac=0.5 
            budget=$(echo "${n_arms}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)
            prob_distro=food_rescue_two_timescale
            reward_type=probability

            python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0 --budget ${budget} --reward_type ${reward_type} --arm_set_low 0 --arm_set_high 1 --prob_distro ${prob_distro} --out_folder journal_results/two_timestep --n_episodes 5 --episode_len ${episode_len} --discount 0.9999 

        done 
    done 
done 
