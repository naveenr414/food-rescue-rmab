# Baseline runs

for seed in 42 43 44
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER

    n_arms=2
    for n_volunteers in 2 4 16
    do 
        tmux send-keys -t match_${seed} "conda activate food; python large_combined_oracle_bandits.py --seed ${seed} --save_name combined_${n_arms}_${n_volunteers} --n_arms ${n_arms} --volunteers_per_arm ${n_volunteers} | tee ../runs/baseline_matching_seed_${seed}_${n_arms}_${n_volunteers}.txt" ENTER
    done 

    n_arms=4
    n_volunteers=4
    tmux send-keys -t match_${seed} "conda activate food; python large_combined_oracle_bandits.py --seed ${seed} --save_name combined_${n_arms}_${n_volunteers} --n_arms ${n_arms} --volunteers_per_arm ${n_volunteers} | tee ../runs/baseline_matching_seed_${seed}_${n_arms}_${n_volunteers}.txt" ENTER

    n_arms=16
    n_volunteers=2
    tmux send-keys -t match_${seed} "conda activate food; python large_combined_oracle_bandits.py --seed ${seed} --save_name combined_${n_arms}_${n_volunteers} --n_arms ${n_arms} --volunteers_per_arm ${n_volunteers} | tee ../runs/baseline_matching_seed_${seed}_${n_arms}_${n_volunteers}.txt" ENTER
done 

# MCTS Runs

for seed in 42 43 44
do
    tmux new-session -d -s mcts_${seed}
    tmux send-keys -t mcts_${seed} ENTER 
    tmux send-keys -t mcts_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER

    n_arms=2
    for n_volunteers in 2 4 16
    do 
        tmux send-keys -t mcts_${seed} "conda activate food; python large_combined_oracle_bandits.py --seed ${seed} --save_name mcts_${n_arms}_${n_volunteers} --n_arms ${n_arms} --volunteers_per_arm ${n_volunteers} | tee ../runs/baseline_matching_seed_${seed}_${n_arms}_${n_volunteers}.txt" ENTER
    done 

    n_arms=4
    n_volunteers=4
    tmux send-keys -t mcts_${seed} "conda activate food; python large_combined_oracle_bandits.py --seed ${seed} --save_name mcts_${n_arms}_${n_volunteers} --n_arms ${n_arms} --volunteers_per_arm ${n_volunteers} | tee ../runs/baseline_matching_seed_${seed}_${n_arms}_${n_volunteers}.txt" ENTER

    n_arms=16
    n_volunteers=2
    tmux send-keys -t mcts_${seed} "conda activate food; python large_combined_oracle_bandits.py --seed ${seed} --save_name mcts_${n_arms}_${n_volunteers} --n_arms ${n_arms} --volunteers_per_arm ${n_volunteers} | tee ../runs/baseline_matching_seed_${seed}_${n_arms}_${n_volunteers}.txt" ENTER
done 

