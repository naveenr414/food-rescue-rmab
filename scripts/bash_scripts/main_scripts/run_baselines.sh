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
        for volunteers in 4 5 10 25 100
        do 
            for budget_frac in 0.25 0.5 0.75 1
            do 
                budget=$(echo "${volunteers}*${budget_frac}" | bc)
                budget=$(printf "%.0f" $budget)
                echo "Volunteers ${volunteers} Budget ${budget}"
                tmux send-keys -t match_${session} "conda activate food; python pure_rl.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type max --arm_set_low 0 --arm_set_high 1 --out_folder baselines/all" ENTER
            done 
        done 
    done 
done 
