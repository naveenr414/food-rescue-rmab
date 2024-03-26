for seed in 42 43 44
do
    tmux new-session -d -s replication_${seed}
    tmux send-keys -t replication_${seed} ENTER 
    tmux send-keys -t replication_${seed} "conda activate food; python replication.py --seed ${seed} --save_name synthetic | tee ../runs/replication_output_${seed}.txt" ENTER
done 