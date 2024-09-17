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

        for recovery_rate in 0 0.001 0.01 0.1 0.25 0.5 1
        do 
            for prob_distro in multi_state_constant multi_state_decreasing multi_state_increasing
            do 
                for n_arms in 4
                do 
                    budget_frac=0.5 
                    budget=$(echo "${n_arms}*${budget_frac}" | bc)
                    budget=$(printf "%.0f" $budget)

                    prob_distro=${prob_distro}
                    reward_type=probability

                    echo ${recovery_rate} ${prob_distro} ${seed}
                    tmux send-keys -t match_${session} "conda activate food; python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --budget ${budget} --reward_type ${reward_type} --discount 0.9999 --arm_set_low 0 --arm_set_high 1 --prob_distro ${prob_distro} --episode_len 1250 --n_episodes 5 --lamb 0 --out_folder journal_results/multi_state --recovery_rate ${recovery_rate} 2>&1 | tee ../../runs/multi_state/${seed}_${recovery_rate}_${prob_distro}.txt" ENTER
                done 
            done 
        done 
    done 
done 
