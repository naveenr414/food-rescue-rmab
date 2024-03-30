# MCTS Runs

for seed in 43 44 45
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts/notebooks" ENTER
    
    tmux send-keys -t match_${seed} "conda activate food; python mcts_ablation.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_ablation.py --seed ${seed} --volunteers_per_arm 2 --n_arms 5 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_ablation.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_ablation.py --seed ${seed} --volunteers_per_arm 2 --n_arms 10 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_ablation.py --seed ${seed} --volunteers_per_arm 10 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_ablation.py --seed ${seed} --volunteers_per_arm 10 --n_arms 10 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER

    for budget in 3 5 8 10
    do 
        echo "Running Budget ${budget}"
        tmux send-keys -t match_${seed} "conda activate food; python mcts_ablation.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget ${budget} --prob_distro uniform" ENTER
    done
done 