# Index runs

for seed in 42 43 44
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER
    
    for budget in 3 10 100 1000
    do 
        echo "Running Budget ${budget}"
        tmux send-keys -t match_${seed} "conda activate food; python real_experiments.py --seed ${seed} --budget ${budget}" ENTER
    done
done 
