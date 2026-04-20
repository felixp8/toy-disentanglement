#!/bin/bash -l

conda activate hw

python train.py -m hydra/launcher=basic \
    'embedding=5d_relu_128x2_noiseless' \
    'dataset=unbiased_tanh' \
    '+dataset.correlation=0.6' \
    'dataset.num_tasks=6,10,20' \
    'model.hidden_dims=[256],[256,256],[256,256,256],[256,256,256,256]' \
    'model.activation=relu' \
    'model.output_activation=none' \
    'wandb=true' \
    'sweep_id=12'