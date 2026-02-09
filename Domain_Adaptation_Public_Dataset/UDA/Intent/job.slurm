#!/bin/bash
#SBATCH --cluster=genius
#SBATCH --account=lp_tearsai
#SBATCH --job-name=training2
#SBATCH --error=test1lr-2.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_p100
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

export CUDA_LAUNCH_BLOCKING=1

# Ativa o ambiente Python correto
source /data/leuven/379/vsc37951/Thesis_env/bin/activate

# Executa o script
python -u /data/leuven/379/vsc37951/Projects/Done/UDA/Intent/train.py --data /data/leuven/379/vsc37951/Projects
