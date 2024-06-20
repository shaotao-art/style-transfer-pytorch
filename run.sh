#!/bin/bash
device_idx=5
config_file=./configs/config.py
run_name=100k-steps-s-w-10.0

CUDA_VISIBLE_DEVICES=$device_idx  python run.py \
                        --config $config_file\
                        --run_name $run_name \
