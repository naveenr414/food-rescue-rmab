# Cohort Size impact
for cohort_size in 4 8 12 16
do 
    tmux new-session -d -s cohort_${cohort_size}
    tmux send-keys -t cohort_${cohort_size} ENTER 

    for seed in 42 43 44
    do
        tmux send-keys -t cohort_${cohort_size} "conda activate food; python baseline_bandit.py --seed ${seed} --n_arms ${cohort_size} --save_name hyperparameter --use_date | tee ../runs/baseline_output_cohort_${cohort_size}_${seed}.txt" ENTER
    done 

    tmux send-keys -t cohort_${cohort_size} "tmux kill-session -t cohort_${cohort_size}" ENTER
done 

# Budget impact
for budget in 1 2 3 5 7
do 
    tmux new-session -d -s budget_${budget}
    tmux send-keys -t budget_${budget} ENTER 

    for seed in 42 43 44
    do
        tmux send-keys -t budget_${budget} "conda activate food; python baseline_bandit.py --seed ${seed} --budget ${budget} --save_name hyperparameter --use_date | tee ../runs/baseline_output_budget_${budget}_${seed}.txt" ENTER
    done 

    tmux send-keys -t budget_${budget} "tmux kill-session -t budget_${budget}" ENTER
done 


