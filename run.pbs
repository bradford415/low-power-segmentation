#!/bin/bash

#PBS -N max-low-power
#PBS -q viprgs -l select=1:ncpus=20:ngpus=2:mem=64gb:gpu_model=a100,walltime=6:00:00
#PBS -j oe

source activate low-power

cd /scratch1/bselee/low-power-segmentation

python train.py --cfg configs/deeplabv3/deeplabv3_lpcvc_palmetto.yaml
