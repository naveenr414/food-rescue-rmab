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
        for volunteers in 15 20 
        do 
            echo ${volunteers}
            budget_frac=0.25 
            budget=$(echo "${volunteers}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)
            tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type max --arm_set_low 0 --arm_set_high 1 --out_folder reward_variation/max_reward" ENTER
        done 

        for time_limit in 0.01 0.1 1 10 
        do 
            tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms 1000 --lamb 0.5 --budget 100 --reward_type max --time_limit ${time_limit} --arm_set_low 0 --arm_set_high 1 --out_folder reward_variation/max_reward" ENTER
        done 
    done 
done 
