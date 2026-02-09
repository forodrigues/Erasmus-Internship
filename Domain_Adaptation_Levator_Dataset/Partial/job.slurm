#!/bin/bash
#SBATCH --cluster=genius
#SBATCH --account=lp_tearsai
#SBATCH --job-name=training2
#SBATCH --output=Final_dn_lowerbound.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_p100
#SBATCH --time=01:00:00
#SBATCH --mem=4G

# Ativa o ambiente Python correto
source /data/leuven/379/vsc37951/Thesis_env/bin/activate

# Executa o script
python -u /data/leuven/379/vsc37951/Projects/Final/Partial/train.py --data /data/leuven/379/vsc37951/Projects
