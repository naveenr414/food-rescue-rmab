# Normal Distribution Runs
for seed in 42 43 44
do
    tmux new-session -d -s norm_${seed}
    tmux send-keys -t norm_${seed} ENTER 
    tmux send-keys -t norm_${seed} "cd ~/food_rescue_rmab/scripts" ENTER

    for dataset in synthetic fr
    do 
        tmux send-keys -t norm_${seed} "conda activate food; python better_bandit.py --seed ${seed} --dataset ${dataset} --save_name normal --use_date | tee ../runs/normal_output_${seed}_${dataset}.txt" ENTER
    done 
done 
