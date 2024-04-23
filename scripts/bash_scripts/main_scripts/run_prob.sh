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
        for arm_set_low in 0 0.25 0.5 0.75 1.0
        do 
            for arm_set_high in 0 0.25 0.5 0.75 1.0
            do 
                if awk -v low="$arm_set_low" -v high="$arm_set_high" 'BEGIN { exit !(high > low) }'; then
                    tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms 10 --lamb 0.5 --budget 5 --reward_type probability --arm_set_low ${arm_set_low} --arm_set_high ${arm_set_high} --out_folder reward_variation/prob_reward" ENTER
                fi 
            done 
        done 
    done 
done 
