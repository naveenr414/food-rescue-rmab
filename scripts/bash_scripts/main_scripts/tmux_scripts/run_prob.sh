#!/bin/bash 

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ~/projects/food_rescue_rmab/scripts/notebooks" ENTER

    for start_seed in 42 45 48 51 54
    do 
        seed=$((${session}+${start_seed}))
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
                        tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type probability --arm_set_low ${arm_set_low} --arm_set_high ${arm_set_high} --out_folder reward_variation/prob_reward" ENTER
                        tmux send-keys -t match_${session} "conda activate food; python baselines.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type probability --arm_set_low ${arm_set_low} --arm_set_high ${arm_set_high} --out_folder baselines/all" ENTER
                    fi 
                done 
            done 
        done 
    done 
done 
