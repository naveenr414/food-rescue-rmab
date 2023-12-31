# Baseline runs

for seed in 42 43 44
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts" ENTER

    for n_arms in 2 4 8 16
    do 
        for n_volunteers in 2 4 16
        do 
            tmux send-keys -t match_${seed} "conda activate food; python large_combined_oracle_bandits.py --seed ${seed} --save_name combined_${n_arms}_${n_volunteers} --n_arms ${n_arms} --volunteers_per_arm ${n_volunteers} | tee ../runs/baseline_matching_seed_${seed}_${n_arms}_${n_volunteers}.txt" ENTER
        done 
    done 
done 

