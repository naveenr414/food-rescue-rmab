# Baseline runs

for seed in 42 43 44
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/food_rescue_rmab/scripts" ENTER

    for n_arms in 4 5 6 7 8
    do 
        tmux send-keys -t match_${seed} "conda activate food; python uncertain_matching_bandits.py --seed ${seed} --match_prob 0.5 --save_name results_${n_arms} --n_arms ${n_arms} | tee ../runs/matching_seed_${seed}_${n_arms}.txt" ENTER
    done 
done 

# # Impact of parameter p
# for p in 0.1 0.25 0.5 0.75 0.9
# do 
#     sanitized_p="${p//./_}"
#     tmux new-session -d -s p_${sanitized_p}
#     tmux send-keys -t p_${sanitized_p} ENTER 

#     for seed in 42 43 44
#     do
#         tmux send-keys -t p_${sanitized_p} "conda activate food; python uncertain_matching_bandits.py --seed ${seed} --match_prob ${p} --save_name p_val --use_date | tee ../runs/p_val_output_${sanitized_p}.txt" ENTER
#     done 

#     # tmux send-keys -t p_${sanitized_p} "tmux kill-session -t p_${sanitized_p}" ENTER
# done 

