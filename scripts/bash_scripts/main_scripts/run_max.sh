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
        for arm_set_high in 0.5 1 2 5 10 20
        do 
            tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms 10 --lamb 0.5 --budget 5 --reward_type max --arm_set_low 0 --arm_set_high ${arm_set_high} --out_folder reward_variation/max_reward" ENTER
        done 
    done 
done 
