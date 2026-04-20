#!/bin/bash -l

conda activate hw

python train.py -m hydra/launcher=basic \
    'embedding=5d_relu_128x2_noiseless' \
    'dataset=biased_wavelet' \
    'dataset.num_tasks=20' \
    'model.hidden_dims=[256],[256,256],[256,256,256],[256,256,256,256]' \
    'model.activation=relu' \
    'model.output_activation=none' \
    'wandb=true' \
    'sweep_id=14'