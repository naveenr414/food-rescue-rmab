#!/bin/bash 

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ~/projects/food_rescue_rmab/scripts/notebooks" ENTER

    for start_seed in 42
    do 
        seed=$((${session}+${start_seed}))
        echo ${seed}
        for universe_size in 10 # 20 50 100
        do 
            for arm_factor in 0.125 # 0.25 0.5 
            do 
                arm_size=$(echo "${universe_size}*${arm_factor}" | bc)
                arm_size=$(printf "%.0f" $arm_size)
                tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms 10 --lamb 0.5 --budget 5 --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --out_folder reward_variation/subset_reward" ENTER
            done 
        done 
    done 
done 
