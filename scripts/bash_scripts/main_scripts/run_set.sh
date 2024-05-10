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
        for n_arms in 4 10 
        do 
            budget_frac=0.5 
            budget=$(echo "${n_arms}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)
            for universe_size in 10 20
            do 
                for arm_factor in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 
                do 
                    arm_size=$(echo "${universe_size}*${arm_factor}" | bc)
                    arm_size=$(printf "%.0f" $arm_size)
                    tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --out_folder reward_variation/subset_reward" ENTER
                    tmux send-keys -t match_${session} "conda activate food; python pure_rl.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --out_folder baselines/all" ENTER
                done 
            done 
        done 
    done 
done 
