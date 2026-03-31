#!/bin/bash
#SBATCH -J teflow
#SBATCH --mem 500GB
#SBATCH --gres gpu:10
#SBATCH --cpus-per-task 48
#SBATCH --constrain "galadriel|eowyn"
#SBATCH --output /Midgard/home/qingwen/logs/teflow/%J.out
#SBATCH --error  /Midgard/home/qingwen/logs/teflow/%J.err

PYTHON=/Midgard/home/qingwen/miniforge3/envs/seflow/bin/python
cd /Midgard/home/qingwen/workspace/OpenSceneFlow

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Midgard/home/qingwen/miniforge3/lib

SOURCE="/local_storage/datasets/qingwen/data/h5py/av2"
$PYTHON train.py slurm_id=$SLURM_JOB_ID wandb_mode=online wandb_project_name=teflow train_data=${SOURCE}/train val_data=${SOURCE}/val \
     model=deltaflow save_top_model=2 val_every=3 train_aug=True voxel_size="[0.15, 0.15, 0.15]" point_cloud_range="[-38.4, -38.4, -3, 38.4, 38.4, 3]" num_workers=16 \
     epochs=15 optimizer.lr=2e-3 batch_size=2 num_frames=5 "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" \
     +ssl_label=seflow_auto loss_fn=teflowLoss optimizer.name=Adam +optimizer.scheduler.name=StepLR +optimizer.scheduler.step_size=8 +optimizer.scheduler.gamma=0.5

# SOURCE="/local_storage/datasets/qingwen/data/h5py/waymo"
# $PYTHON train.py slurm_id=$SLURM_JOB_ID wandb_mode=online wandb_project_name=teflow train_data=${SOURCE}/train val_data=${SOURCE}/valid \
#      model=deltaflow save_top_model=2 val_every=3 train_aug=True voxel_size="[0.15, 0.15, 0.15]" point_cloud_range="[-38.4, -38.4, -3, 38.4, 38.4, 3]" num_workers=16 \
#      epochs=15 optimizer.lr=2e-3 batch_size=2 num_frames=5 "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" \
#      +ssl_label=seflow_auto loss_fn=teflowLoss optimizer.name=Adam +optimizer.scheduler.name=StepLR +optimizer.scheduler.step_size=8 +optimizer.scheduler.gamma=0.5

# SOURCE="/local_storage/datasets/qingwen/data/h5py/nus"
# $PYTHON train.py slurm_id=$SLURM_JOB_ID wandb_mode=online wandb_project_name=teflow train_data=${SOURCE}/train val_data=${SOURCE}/val \
#      model=deltaflow save_top_model=2 val_every=3 train_aug=True voxel_size="[0.15, 0.15, 0.15]" point_cloud_range="[-38.4, -38.4, -3, 38.4, 38.4, 3]" num_workers=16 \
#      epochs=15 optimizer.lr=2e-3 batch_size=2 num_frames=5 "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" \
#      +ssl_label=seflow_auto loss_fn=teflowLoss optimizer.name=Adam +optimizer.scheduler.name=StepLR +optimizer.scheduler.step_size=8 +optimizer.scheduler.gamma=0.5
