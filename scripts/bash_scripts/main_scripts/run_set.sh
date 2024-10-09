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
        for universe_size in 20
        do 
            for arm_factor in 0.3
            do 
                for prob_distro in uniform 
                do 
                    arm_size=$(echo "${universe_size}*${arm_factor}" | bc)
                    arm_size=$(printf "%.0f" $arm_size)
                    arm_set_high=$((arm_size + 2))
                    python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward
                    python baselines.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder baselines/all
                done 
            done 
        done 

        universe_size=20
        arm_set_high=3
        arm_size=1
        prob_distro=linearity
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward

        arm_size=2
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward

        arm_set_high=4
        arm_size=1
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward

        arm_size=2
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward

        arm_size=3
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward

        arm_set_high=5
        arm_size=1
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward

        arm_size=2
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward

        arm_size=3
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward

        arm_size=4
        python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward
    done 
done 
