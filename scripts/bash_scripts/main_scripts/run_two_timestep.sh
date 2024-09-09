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

        for n_arms in 4 10 25 100
        do 
            budget_frac=0.5 
            budget=$(echo "${n_arms}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)

            for prob_distro in food_rescue_two_timestep food_rescue_two_timestep_quick
            do 
                reward_type=probability_two_timestep

                tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type ${reward_type} --arm_set_low 0 --arm_set_high 1 --prob_distro ${prob_distro} --out_folder journal_results/two_timestep --n_episodes 5 --episode_len 50000" ENTER
            done 
        done 
    done 
done 
