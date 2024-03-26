# Index runs

for seed in 43 44 45
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER
    
    tmux send-keys -t match_${seed} "conda activate food; python baseline_real_experiments.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python baseline_real_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python baseline_real_experiments.py --seed ${seed} --volunteers_per_arm 2 --n_arms 5 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python baseline_real_experiments.py --seed ${seed} --volunteers_per_arm 10 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python baseline_real_experiments.py --seed ${seed} --volunteers_per_arm 2 --n_arms 10 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python baseline_real_experiments.py --seed ${seed} --volunteers_per_arm 10 --n_arms 10 --lamb 0.5 --budget 3 --prob_distro uniform" ENTER


    for budget in 3 5 8 10
    do 
        echo "Running Budget ${budget}"
        tmux send-keys -t match_${seed} "conda activate food; python baseline_real_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget ${budget} --prob_distro uniform" ENTER
    done

    for lamb in 0 0.25 0.5 0.75 1
    do 
        echo "Lamb ${lamb}"
        tmux send-keys -t match_${seed} "conda activate food; python baseline_real_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb ${lamb} --budget 3 --prob_distro uniform" ENTER
    done  

    for prob_distro in uniform uniform_small uniform_large normal 
    do 
        echo "Prob Distro ${prob_distro}"
        tmux send-keys -t match_${seed} "conda activate food; python baseline_real_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro ${prob_distro}" ENTER
    done  
done 

# MCTS Runs

for seed in 43 44 45
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER
    
    tmux send-keys -t match_${seed} "conda activate food; python mcts_real_experiments.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr  5e-4 --value_lr 1e-4 --n_episodes 200 --train_iterations 30 --test_iterations 30" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_real_experiments.py --seed ${seed} --volunteers_per_arm 2 --n_arms 5 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr  5e-4 --value_lr 1e-4 --n_episodes 200 --train_iterations 30 --test_iterations 30" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_real_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr  5e-3 --value_lr 1e-4 --n_episodes 300 --train_iterations 30 --test_iterations 30" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_real_experiments.py --seed ${seed} --volunteers_per_arm 2 --n_arms 10 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr  5e-3 --value_lr 1e-4 --n_episodes 300 --train_iterations 30 --test_iterations 30" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_real_experiments.py --seed ${seed} --volunteers_per_arm 10 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr  5e-3 --value_lr 1e-4 --n_episodes 200 --train_iterations 30 --test_iterations 30" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_real_experiments.py --seed ${seed} --volunteers_per_arm 10 --n_arms 10 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr  5e-3 --value_lr 1e-4 --n_episodes 300 --train_iterations 30 --test_iterations 30" ENTER

    for budget in 3 5 8 10
    do 
        echo "Running Budget ${budget}"
        tmux send-keys -t match_${seed} "conda activate food; python mcts_real_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget ${budget} --prob_distro uniform --policy_lr  5e-3 --value_lr 1e-4 --n_episodes 300 --train_iterations 30 --test_iterations 30" ENTER
    done

    for lamb in 0 0.25 0.5 0.75 1
    do 
        echo "Lamb ${lamb}"
        tmux send-keys -t match_${seed} "conda activate food; python mcts_real_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb ${lamb} --budget 3 --prob_distro uniform --policy_lr  5e-3 --value_lr 1e-4 --n_episodes 300 --train_iterations 30 --test_iterations 30" ENTER
    done  

    for prob_distro in uniform uniform_small uniform_large normal 
    do 
        echo "Prob Distro ${prob_distro}"
        tmux send-keys -t match_${seed} "conda activate food; python mcts_real_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro ${prob_distro} --policy_lr  5e-3 --value_lr 1e-4 --n_episodes 300 --train_iterations 30 --test_iterations 30" ENTER
    done  
done 


