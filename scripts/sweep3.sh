#!/bin/bash -l

conda activate hw

python train.py -m hydra/launcher=basic \
    'embedding=5dhs_relu_128x2_noiseless' \
    'dataset.num_tasks=2,4,6,8,10,15,20,30,50' \
    'model.hidden_dims=[256],[256,256],[256,256,256],[256,256,256,256]' \
    'model.activation=relu' \
    'wandb=true' \
    'sweep_id=3'