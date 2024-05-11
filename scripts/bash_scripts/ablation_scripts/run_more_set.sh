#!/bin/bash 

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ~/projects/food_rescue_rmab/scripts/notebooks" ENTER

    for start_seed in 42 45 48
    do 
        seed=$((${session}+${start_seed}))
        echo ${seed}
        for n_arms in 25 100
        do 
            budget_frac=0.5 
            budget=$(echo "${n_arms}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)
            for universe_size in 20
            do 
                for arm_factor in 0.3
                do 
                    for prob_distro in high_prob uniform 
                    do 
                        arm_size=$(echo "${universe_size}*${arm_factor}" | bc)
                        arm_size=$(printf "%.0f" $arm_size)
                        arm_set_high=$((arm_size + 2))
                        tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder reward_variation/subset_reward" ENTER
                    done 
                done 
            done 
        done 
    done 
done 
