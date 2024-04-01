# Index runs

for seed in 43 44 45
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts/notebooks" ENTER

    tmux send-keys -t match_${seed} "conda activate food; python pure_mcts.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0.5 --budget 3" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python pure_mcts.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python pure_mcts.py --seed ${seed} --volunteers_per_arm 2 --n_arms 5 --lamb 0.5 --budget 3" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python pure_mcts.py --seed ${seed} --volunteers_per_arm 10 --n_arms 2 --lamb 0.5 --budget 3" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python pure_mcts.py --seed ${seed} --volunteers_per_arm 2 --n_arms 10 --lamb 0.5 --budget 3" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python pure_mcts.py --seed ${seed} --volunteers_per_arm 10 --n_arms 10 --lamb 0.5 --budget 3" ENTER

    for budget in 5 8 10 
    do 
        tmux send-keys -t match_${seed} "conda activate food; python pure_mcts.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget ${budget}" ENTER
    done 

    for lamb in 0 0.25 0.75 1 
    do 
        tmux send-keys -t match_${seed} "conda activate food; python pure_mcts.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb ${lamb} --budget 3" ENTER
    done 
done 
