#!/bin/bash 

cd scripts/notebooks

for seed in $(seq 43 57); 
do 
    echo ${seed}

    volunteers=100
    budget=25
    python all_policies.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type probability --prob_distro food_rescue --arm_set_low 0 --arm_set_high 1 --out_folder food_rescue_policies
    python baselines.py --seed ${seed} --volunteers_per_arm 1 --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type probability --prob_distro food_rescue --arm_set_low 0 --arm_set_high 1 --out_folder baselines/food_rescue_policies

    volunteers=20
    volunteers_per_arm=1
    budget=10
    python all_policies.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type probability --prob_distro food_rescue_top --arm_set_low 0 --arm_set_high 1 --out_folder food_rescue_policies
    python baselines.py --seed ${seed} --volunteers_per_arm ${volunteers_per_arm} --n_arms ${volunteers} --lamb 0.5 --budget ${budget} --reward_type probability --prob_distro food_rescue_top --arm_set_low 0 --arm_set_high 1 --out_folder baselines/food_rescue_policies 
done 
