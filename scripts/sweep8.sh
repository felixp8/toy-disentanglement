#!/bin/bash -l

conda activate hw

python train.py -m hydra/launcher=basic \
    'embedding=5d_relu_128x2_noiseless' \
    'dataset=unbiased_sparse_tanh' \
    '+dataset.correlation=0.5' \
    'dataset.num_tasks=2,4,6,8,10,15,20,30,50' \
    'model.hidden_dims=[256],[256,256],[256,256,256],[256,256,256,256]' \
    'model.activation=relu' \
    'wandb=true' \
    'sweep_id=8'