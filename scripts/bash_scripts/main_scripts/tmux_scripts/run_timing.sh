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
        for volunteers in 5 10 15 20 25 
        do 
            echo ${volunteers}
            budget_frac=0.5
            budget=$(echo "${volunteers}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)
            tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type set_cover --arm_set_low 6 --arm_set_high 8 --universe_size 20 --run_rate_limits --out_folder reward_variation/subset_reward" ENTER
        done 
    done 
done 
