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

        volunteers=100
        budget=25
        tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type probability --prob_distro food_rescue --arm_set_low 0 --arm_set_high 1 --out_folder food_rescue_policies" ENTER
        tmux send-keys -t match_${session} "conda activate food; python baselines.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type probability --prob_distro food_rescue --arm_set_low 0 --arm_set_high 1 --out_folder baselines/food_rescue_policies" ENTER

        volunteers=20
        volunteers_per_arm=1
        budget=10
        tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type probability --prob_distro food_rescue_top --arm_set_low 0 --arm_set_high 1 --out_folder food_rescue_policies" ENTER
        tmux send-keys -t match_${session} "conda activate food; python baselines.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type probability --prob_distro food_rescue_top --arm_set_low 0 --arm_set_high 1 --out_folder baselines/food_rescue_policies" ENTER
    done 
done 
