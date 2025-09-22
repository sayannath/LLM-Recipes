#!/bin/bash
####### ARC #############
#SBATCH --job-name=gptoss                     # Job name
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1                    # keep all GPUs on one node
#SBATCH --gres=gpu:3                 # request 3 GPUs
#SBATCH --mem=128G
#SBATCH --time=23:59:59
#SBATCH --ntasks-per-node=3          # 1 task per GPU
#SBATCH --cpus-per-task=8            # tune per your cluster
#SBATCH --output=my_gpu_job_%j.log

####### Load necessary modules #############
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_env

####### Run the script #############
cd ..
cd finetune/
python finetune_gpt_oss.py