#!/bin/bash -l

conda activate hw

python train.py -m \
    'latent_dim=5' \
    'num_tasks=1,2,3,4,5,6,7,8,9,10' \
    'model.hidden_dims=[128],[128,128],[128,128,128],[128,128,128,128],[256],[256,256],[256,256,256],[256,256,256,256]' \
    'model.activation=relu,tanh,gelu' \
    'wandb=true' \
    'dataset=biased_classification'