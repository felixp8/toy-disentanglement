#!/bin/bash -l

conda activate hw

python train.py -m \
    'embedding=5d_relu_128x2_noiseless,5d_relu_128x4_noiseless,5d_tanh_128x2_noiseless' \
    'num_tasks=1,2,3,4,5,6,7,8,9,10' \
    'model.hidden_dims=[128],[128,128],[128,128,128],[128,128,128,128],[256],[256,256],[256,256,256],[256,256,256,256]' \
    'model.activation=relu,tanh,gelu' \
    'wandb=true'