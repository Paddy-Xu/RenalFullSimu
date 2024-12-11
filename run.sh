#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=38000M
##SBATCH -p gpu --gres=gpu:titanrtx:2
##SBATCH -p gpu --gres=gpu:A100:2

#SBATCH --time=68:00:00
#your script, in this case: write the hostname and the ids of the chosen gpus.
echo $CUDA_VISIBLE_DEVICES

python tree_model.py

#python pathology_simu.py
