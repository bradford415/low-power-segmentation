#!/bin/bash

python main.py --train \
               --experiment bn_lr7e-3 \
               --backbone resnet101 \
               --dataset pascal \
               --base_lr 0.007 \
                
