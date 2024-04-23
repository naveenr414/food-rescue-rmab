#!/bin/bash 

for session in 1 2 3
do
    tmux new-session -d -s match_${session}
    tmux send-keys -t match_${session} ENTER 
    tmux send-keys -t match_${session} "cd ~/projects/food_rescue_rmab/scripts/notebooks" ENTER

    for start_seed in 0 3 6
    do 
        seed=$((${session}+${start_seed}))
        echo ${seed}
        for volunteers_per_arm in 2 5 
        do 
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 10 --arm_set_low 1 --arm_set_high 5" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 10 --arm_set_low 2 --arm_set_high 5" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 10 --arm_set_low 4 --arm_set_high 5" ENTER
            
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 10 --arm_set_low 1 --arm_set_high 2" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 10 --arm_set_low 2 --arm_set_high 2" ENTER

            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 20 --arm_set_low 4 --arm_set_high 10" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 20 --arm_set_low 8 --arm_set_high 10" ENTER

            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 20 --arm_set_low 1 --arm_set_high 5" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 20 --arm_set_low 2 --arm_set_high 5" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 20 --arm_set_low 4 --arm_set_high 5" ENTER
            
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 20 --arm_set_low 1 --arm_set_high 2" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 0 --budget 2 --universe_size 20 --arm_set_low 2 --arm_set_high 2" ENTER

            
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm 2 --n_arms 5 --lamb 0.5 --budget 3" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm 10 --n_arms 2 --lamb 0.5 --budget 3" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm 2 --n_arms 10 --lamb 0.5 --budget 3" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm 10 --n_arms 10 --lamb 0.5 --budget 3" ENTER
        done 

        for budget in 1 2 3 4
        do 
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0 --budget ${budget}" ENTER
        done 

        for budget in 2 4 6 8 10
        do 
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0 --budget ${budget}" ENTER
        done 

        for lamb in 0 0.25 0.5 0.75 1 
        do 
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb ${lamb} --budget 2" ENTER
            tmux send-keys -t match_${session} "conda activate food; python mcts_shapley.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb ${lamb} --budget 4" ENTER
        done 
    done 
done 
