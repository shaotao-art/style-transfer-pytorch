#!/bin/bash
device_idx=1
max_steps=800
config_file=./configs/config.py

lrs=(3e-4 5e-5 1e-3)
for lr in "${lrs[@]}"; do
    run_name=find-lr-$lr
    CUDA_VISIBLE_DEVICES=$device_idx  python run.py \
                            --config $config_file\
                            --run_name $run_name \
                            --find_lr \
                            --max_steps $max_steps \
                            --lr $lr
done
