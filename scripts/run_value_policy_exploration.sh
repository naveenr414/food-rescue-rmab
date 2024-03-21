# Policy Only runs

for seed in 43 44 45
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER
    
    tmux send-keys -t match_${seed} "conda activate food; python mcts_semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr 1e-4 --n_episodes 200 --train_iterations 30 --test_iterations 0 --out_folder value_policy_exploration/policy_only" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr 1e-4 --n_episodes 200 --train_iterations 30 --test_iterations 0 --out_folder value_policy_exploration/policy_only" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 2 --n_arms 5 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr 1e-4 --n_episodes 200 --train_iterations 30 --test_iterations 0 --out_folder value_policy_exploration/policy_only" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 10 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr 1e-4 --n_episodes 200 --train_iterations 30 --test_iterations 0 --out_folder value_policy_exploration/policy_only" ENTER
    tmux send-keys -t match_${seed} "conda activate food; python mcts_semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 2 --n_arms 10 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr 1e-4 --n_episodes 200 --train_iterations 30 --test_iterations 0 --out_folder value_policy_exploration/policy_only" ENTER

done 

# Value + Policy Exploration
for seed in 43 44 45
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER
    
    for policy_lr in 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 
    do 
        tmux send-keys -t match_${seed} "conda activate food; python value_policy_exploration.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr ${policy_lr} --value_lr 1e-4 --n_episodes 200 --train_iterations 30 --test_iterations 30 --out_folder value_policy_exploration/exploration" ENTER
        tmux send-keys -t match_${seed} "conda activate food; python value_policy_exploration.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr ${policy_lr} --value_lr 1e-4 --n_episodes 200 --train_iterations 30 --test_iterations 30 --out_folder value_policy_exploration/exploration" ENTER
    done 

    for value_lr in 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 
    do 
        tmux send-keys -t match_${seed} "conda activate food; python value_policy_exploration.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr ${value_lr} --n_episodes 200 --train_iterations 30 --test_iterations 30 --out_folder value_policy_exploration/exploration" ENTER
        tmux send-keys -t match_${seed} "conda activate food; python value_policy_exploration.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr ${value_lr}--n_episodes 200 --train_iterations 30 --test_iterations 30 --out_folder value_policy_exploration/exploration" ENTER
    done 

    for train_iterations in 10 20 30 40 50 60
    do 
        tmux send-keys -t match_${seed} "conda activate food; python value_policy_exploration.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr 1e-4 --n_episodes 200 --train_iterations ${train_iterations} --test_iterations 30 --out_folder value_policy_exploration/exploration" ENTER
        tmux send-keys -t match_${seed} "conda activate food; python value_policy_exploration.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr 1e-4 --n_episodes 200 --train_iterations ${train_iterations} --test_iterations 30 --out_folder value_policy_exploration/exploration" ENTER
    done 

    for num_episodes in 50 100 200 400
    do 
        tmux send-keys -t match_${seed} "conda activate food; python value_policy_exploration.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr 1e-4 --n_episodes ${num_episodes} --train_iterations 30 --test_iterations 30 --out_folder value_policy_exploration/exploration" ENTER
        tmux send-keys -t match_${seed} "conda activate food; python value_policy_exploration.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --policy_lr 5e-4 --value_lr 1e-4 --n_episodes ${num_episodes} --train_iterations 30 --test_iterations 30 --out_folder value_policy_exploration/exploration" ENTER
    done 
done 



