#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --gres=gpu:1

set -x

export WORK_DIR="/cfarhomes/psando/Documents/UAPs/gd-uap-pytorch/"

srun bash -c "cd ${WORK_DIR} && python3 train.py --model vgg16"
