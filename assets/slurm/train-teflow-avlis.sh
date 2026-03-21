#!/bin/bash
#SBATCH -J teflow
#SBATCH -A NAISS2026-3-96 -p alvis
#SBATCH -N 1 --gpus-per-node=T4:8
#SBATCH -t 5-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /cephyr/users/qingwenz/Alvis/workspace/OpenSceneFlow/logs/slurm/%J.out
#SBATCH --error  /cephyr/users/qingwenz/Alvis/workspace/OpenSceneFlow/logs/slurm/%J.err

cd /cephyr/users/qingwenz/Alvis/workspace/OpenSceneFlow
PYTHON="apptainer run --nv --writable-tmpfs /mimer/NOBACKUP/groups/kthrpl_patric/data/apptainer/opensf-full.sif"

# sometimes diff gpus have different CUDA capability, and the compile package may not working
# PYTHON="/mimer/NOBACKUP/groups/kthrpl_patric/users/qingwenz/miniforge3/envs/opensf/bin/python"

# ===> Need change it data path changed <===
SOURCE="/mimer/NOBACKUP/groups/kthrpl_patric/data/h5py/av2"

# ========================= TeFlow num_frame=5 =========================
$PYTHON train.py slurm_id=$SLURM_JOB_ID wandb_mode=online wandb_project_name=teflow train_data=${SOURCE}/sensor/train val_data=${SOURCE}/sensor/val \
     model=deltaflow save_top_model=2 val_every=3 train_aug=True "voxel_size=[0.15, 0.15, 0.15]" "point_cloud_range=[-38.4, -38.4, -3, 38.4, 38.4, 3]" \
     num_workers=16 epochs=15 batch_size=2 num_frames=5 "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" \
     +ssl_label=seflow_auto loss_fn=teflowLoss optimizer.name=Adam optimizer.lr=2e-3 +optimizer.scheduler.name=StepLR +optimizer.scheduler.step_size=9 +optimizer.scheduler.gamma=0.5

