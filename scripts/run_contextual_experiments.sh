# Index runs

for seed in 42 43 44
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER
    
    for volunteers_per_arm in 5 10 20 50 100
    do 
        echo "Running Volunteers per arm ${volunteer_per_arm}"
        tmux send-keys -t match_${seed} "conda activate food; python contextual_experiments.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 1 --budget 3 --prob_distro uniform" ENTER
    done 

    for n_arms in 2 5 10 20 50 100
    do 
        echo "Running N arms ${n_arms}"
        tmux send-keys -t match_${seed} "conda activate food; python contextual_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms ${n_arms} --lamb 1 --budget 3 --prob_distro uniform" ENTER
    done 

    for budget in 3 5 8 10
    do 
        echo "Running Budget ${budget}"
        tmux send-keys -t match_${seed} "conda activate food; python contextual_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 1 --budget ${budget} --prob_distro uniform" ENTER
    done

    for lamb in 0 0.25 0.5 0.75 1
    do 
        echo "Lamb ${lamb}"
        tmux send-keys -t match_${seed} "conda activate food; python contextual_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb ${lamb} --budget 3 --prob_distro uniform" ENTER
    done  
done 

