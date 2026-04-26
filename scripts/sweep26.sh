#!/bin/bash -l

conda activate hw

python train.py -m hydra/launcher=basic \
    'embedding=5d_relu_128x2_noiseless' \
    'dataset=gp_classification' \
    'dataset.length_scale=1.0' \
    'dataset.num_tasks=5,10,20,30,50' \
    'model.hidden_dims=[256,256,256,256,256]' \
    'model.activation=relu' \
    'model.output_activation=tanh' \
    'wandb=true' \
    'num_epochs=2000' \
    'sweep_id=26'