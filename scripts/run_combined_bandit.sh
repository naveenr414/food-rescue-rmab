# Analyze Pareto Frontier 

for seed in 42 43 44
do
    tmux new-session -d -s combined_${seed}
    tmux send-keys -t combined_${seed} ENTER 
    tmux send-keys -t combined_${seed} "conda activate food; python combined_bandit.py --seed ${seed} --save_name combined_lamb | tee ../runs/combined_seed_${seed}.txt" ENTER
done 