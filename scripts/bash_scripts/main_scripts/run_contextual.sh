#!/bin/bash 

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ~/projects/food_rescue_rmab/scripts/notebooks" ENTER

    for start_seed in  42 45 48 51 54
    do 
        seed=$((${session}+${start_seed}))
        echo ${seed}

        for n_arms in 4 10 
        do 
            budget_frac=0.5 
            budget=$(echo "${n_arms}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)

            prob_distro=food_rescue_context
            reward_type=probability_context

            tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type ${reward_type} --arm_set_low 0 --arm_set_high 1 --prob_distro ${prob_distro} --out_folder journal_results/contextual" ENTER
            # tmux send-keys -t match_${session} "conda activate food; python baselines.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type ${reward_type} --arm_set_low 0 --arm_set_high 1 --prob_distro ${prob_distro} --out_folder baselines/journal" ENTER
        done 
    done 
done 
