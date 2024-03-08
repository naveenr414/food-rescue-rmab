# Index runs

# for seed in 42 43 44
# do
#     tmux new-session -d -s match_${seed}
#     tmux send-keys -t match_${seed} ENTER 
#     tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER
    
#     for volunteers_per_arm in 5 10 20 50 100
#     do 
#         echo "Running Volunteers per arm ${volunteer_per_arm}"
#         tmux send-keys -t match_${seed} "conda activate food; python semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 1 --budget 3 --prob_distro uniform" ENTER
#     done 

#     for n_arms in 2 5 10 20 50 100
#     do 
#         echo "Running N arms ${n_arms}"
#         tmux send-keys -t match_${seed} "conda activate food; python semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms ${n_arms} --lamb 1 --budget 3 --prob_distro uniform" ENTER
#     done 

#     for budget in 3 5 8 10
#     do 
#         echo "Running Budget ${budget}"
#         tmux send-keys -t match_${seed} "conda activate food; python semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 1 --budget ${budget} --prob_distro uniform" ENTER
#     done

#     for lamb in 0 0.5 1 2 4 8
#     do 
#         echo "Lamb ${lamb}"
#         tmux send-keys -t match_${seed} "conda activate food; python semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb ${lamb} --budget 3 --prob_distro uniform" ENTER
#     done  

#     for prob_distro in uniform uniform_small uniform_large normal 
#     do 
#         echo "Prob Distro ${prob_distro}"
#         tmux send-keys -t match_${seed} "conda activate food; python semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 1 --budget 3 --prob_distro ${prob_distro}" ENTER
#     done  
# done 

# MCTS Runs

for seed in 42 43 44
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER
    
    # for volunteers_per_arm in 5 10 20 50 100
    # do 
    #     echo "Running Volunteers per arm ${volunteer_per_arm}"
    #     tmux send-keys -t match_${seed} "conda activate food; python mcts_semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms 2 --lamb 1 --budget 3 --prob_distro uniform" ENTER
    # done 

    for n_arms in 2 5 10 20 50 100
    do 
        echo "Running N arms ${n_arms}"
        tmux send-keys -t match_${seed} "conda activate food; python mcts_semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms ${n_arms} --lamb 1 --budget 3 --prob_distro uniform" ENTER
    done 

    for budget in 3 5 8 10
    do 
        echo "Running Budget ${budget}"
        tmux send-keys -t match_${seed} "conda activate food; python mcts_semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 1 --budget ${budget} --prob_distro uniform" ENTER
    done

    for lamb in 0 0.5 1 2 4 8
    do 
        echo "Lamb ${lamb}"
        tmux send-keys -t match_${seed} "conda activate food; python mcts_semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb ${lamb} --budget 3 --prob_distro uniform" ENTER
    done  

    for prob_distro in uniform uniform_small uniform_large normal 
    do 
        echo "Prob Distro ${prob_distro}"
        tmux send-keys -t match_${seed} "conda activate food; python mcts_semi_synthetic_experiments.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 1 --budget 3 --prob_distro ${prob_distro}" ENTER
    done  
done 


