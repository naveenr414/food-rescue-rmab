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
        for arm_set_high in 1
        do 
            for prob_distro in uniform one_time 
            do 
                python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type max --arm_set_low 0 --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/max_reward
                python baselines.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type max --arm_set_low 0 --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder baselines/all
            done 
        done 
    done 
done 
