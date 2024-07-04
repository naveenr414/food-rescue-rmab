#!/bin/bash 

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ~/projects/food_rescue_preferences/scripts/notebooks" ENTER

    for start_seed in 42 45 48 51 54
    do 
        seed=$((${session}+${start_seed}))
        echo ${seed}
        for n_arms in 4 10 20 50
        do 
            budget_frac=0.5 
            budget=$(echo "${n_arms}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)
            tmux send-keys -t match_${session} "conda activate preferences; python global_transitions.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0 --budget ${budget} --reward_type global_transition --prob_distro global_transition --arm_set_low 0 --arm_set_high 1 --out_folder unknown_transitions/baseline_policies" ENTER
        done 

        n_arms=20
        for budget_frac in 0.25 0.5 0.75
        do 
            budget=$(echo "${n_arms}*${budget_frac}" | bc)
            budget=$(printf "%.0f" $budget)
            tmux send-keys -t match_${session} "conda activate preferences; python global_transitions.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0 --budget ${budget} --reward_type global_transition --prob_distro global_transition --arm_set_low 0 --arm_set_high 1 --out_folder unknown_transitions/baseline_policies" ENTER
        done 

        n_arms=20
        budget=5
        for prob_distro in global_transition global_transition_extreme global_transition_high_match global_transition_high_match_impact
        do 
            tmux send-keys -t match_${session} "conda activate preferences; python global_transitions.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${n_arms} --lamb 0 --budget ${budget} --reward_type global_transition --prob_distro ${prob_distro} --arm_set_low 0 --arm_set_high 1 --out_folder unknown_transitions/baseline_policies" ENTER
        done 
    done 
done 
