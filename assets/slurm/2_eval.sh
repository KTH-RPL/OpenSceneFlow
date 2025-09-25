#!/bin/bash
#SBATCH -J eval
#SBATCH --gpus 1
#SBATCH -t 01:00:00
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/seflow/logs/slurm/%J_eval.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/seflow/logs/slurm/%J_eval.err


PYTHON=/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/opensf/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib
cd /proj/berzelius-2023-364/users/x_qinzh/workspace/OpenSceneFlow


# ====> leaderboard model
# $PYTHON eval.py wandb_mode=online dataset_path=/proj/berzelius-2023-364/users/x_qinzh/data/av2/autolabel av2_mode=test \
#     checkpoint=/proj/berzelius-2023-154/users/x_qinzh/seflow/logs/wandb/seflow-10086990/checkpoints/epoch_19_seflow.ckpt \
#     save_res=True

$PYTHON eval.py wandb_mode=online dataset_path=/proj/berzelius-2023-364/users/x_qinzh/data/av2/autolabel av2_mode=val \
    checkpoint=/proj/berzelius-2023-154/users/x_qinzh/seflow/logs/wandb/seflow-10086990/checkpoints/epoch_19_seflow.ckpt