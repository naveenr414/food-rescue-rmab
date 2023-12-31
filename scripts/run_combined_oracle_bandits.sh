# Baseline runs

for seed in 42 43 44
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER

    for n_arms in 4 5
    do 
        tmux send-keys -t match_${seed} "conda activate food; python combined_oracle_bandits.py --seed ${seed} --match_prob 0.5 --save_name heterogeneous_arms_${n_arms} --n_arms ${n_arms} | tee ../runs/baseline_matching_seed_${seed}_${n_arms}.txt" ENTER
    done 
done 

