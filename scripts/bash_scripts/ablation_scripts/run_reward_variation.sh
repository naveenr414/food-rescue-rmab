# Index runs

for seed in 43 44 45
do
    tmux new-session -d -s match_${seed}
    tmux send-keys -t match_${seed} ENTER 
    tmux send-keys -t match_${seed} "cd ~/projects/food_rescue_rmab/scripts/notebooks" ENTER

    for power in 0.1 0.25 0.5 0.75 1 
    do 
        tmux send-keys -t match_${seed} "conda activate food; python reward_variation.py --seed ${seed} --volunteers_per_arm 2 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --power ${power} --out_folder reward_variation" ENTER
        tmux send-keys -t match_${seed} "conda activate food; python reward_variation.py --seed ${seed} --volunteers_per_arm 5 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --power ${power} --out_folder reward_variation" ENTER
        tmux send-keys -t match_${seed} "conda activate food; python reward_variation.py --seed ${seed} --volunteers_per_arm 2 --n_arms 5 --lamb 0.5 --budget 3 --prob_distro uniform --power ${power} --out_folder reward_variation" ENTER
        tmux send-keys -t match_${seed} "conda activate food; python reward_variation.py --seed ${seed} --volunteers_per_arm 10 --n_arms 2 --lamb 0.5 --budget 3 --prob_distro uniform --power ${power} --out_folder reward_variation" ENTER
        tmux send-keys -t match_${seed} "conda activate food; python reward_variation.py --seed ${seed} --volunteers_per_arm 2 --n_arms 10 --lamb 0.5 --budget 3 --prob_distro uniform --power ${power} --out_folder reward_variation" ENTER
        tmux send-keys -t match_${seed} "conda activate food; python reward_variation.py --seed ${seed} --volunteers_per_arm 10 --n_arms 10 --lamb 0.5 --budget 3 --prob_distro uniform --power ${power} --out_folder reward_variation" ENTER
    done 
done 
