#!/bin/bash
####### ARC #############
#SBATCH --job-name=ft-roberta                 # Job name
#SBATCH --partition=gpu-l40                   # Partition name
#SBATCH --mem=64G                             # Memory total in GB (for all cores)
#SBATCH --time=23:59:59                       # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                          # Request GPU resource
#SBATCH --output=my_gpu_job_%j.log            # Standard output and error log

####### Load necessary modules #############
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_env

####### Run the script #############
cd ..
cd finetune/
python finetune_roberta.py