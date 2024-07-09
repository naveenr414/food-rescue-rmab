#!/bin/bash 

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ~/projects/food_rescue_preferences/scripts/notebooks" ENTER

    for start_seed in 42 45
    do 
        seed=$((${session}+${start_seed}))
        echo ${seed}
        for n_arms in 4 10 
        do 
            budget_frac=0.5 
            budget=$(echo "${n_arms}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)
            for universe_size in 20
            do 
                for arm_factor in 0.3
                do 
                    for prob_distro in uniform 
                    do 
                        arm_size=$(echo "${universe_size}*${arm_factor}" | bc)
                        arm_size=$(printf "%.0f" $arm_size)
                        arm_set_high=$((arm_size + 2))
                        tmux send-keys -t match_${session} "conda activate preferences; python unknown_parameters.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type set_cover --universe_size ${universe_size} --arm_set_low ${arm_size} --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder unknown_parameters" ENTER
                    done 
                done 
            done
        done
        
        for n_arms in 4 10 
        do 
            budget_frac=0.5 
            budget=$(echo "${n_arms}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)
            tmux send-keys -t match_${session} "conda activate preferences; python unknown_parameters.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type linear --arm_set_low 0 --arm_set_high 1 --out_folder unknown_parameters" ENTER
        done 
        
        for n_arms in 4 10 
        do 
            budget_frac=0.5 
            budget=$(echo "${n_arms}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)
            for arm_set_high in 1
            do 
                for prob_distro in uniform one_time 
                do 
                    tmux send-keys -t match_${session} "conda activate preferences; python unknown_parameters.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type max --arm_set_low 0 --arm_set_high ${arm_set_high} --prob_distro ${prob_distro} --out_folder unknown_parameters" ENTER
                done 
            done 
        done

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
                        tmux send-keys -t match_${session} "conda activate preferences; python unknown_parameters.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0.5 --budget ${budget} --reward_type probability --arm_set_low ${arm_set_low} --arm_set_high ${arm_set_high} --out_folder unknown_parameters" ENTER
                    fi 
                done 
            done 
        done   
    done 
done 
