#!/bin/bash
#SBATCH -J dataprocess
#SBATCH -p berzelius-cpu
#SBATCH --cpus-per-task 64
#SBATCH --mem 128G
#SBATCH --mincpus=64
#SBATCH -t 2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/OpenSceneFlow/logs/slurm/%J_data.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/OpenSceneFlow/logs/slurm/%J_data.err

PYTHON=/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/sftool/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib
cd /proj/berzelius-2023-154/users/x_qinzh/OpenSceneFlow/
# export HYDRA_FULL_ERROR=1


# =============== Argoverse2
# $PYTHON dataprocess/extract_av2.py --nproc 64 \
#     --av2_type sensor \
#     --data_mode train \
#     --argo_dir /proj/berzelius-2023-154/users/x_qinzh/av2 \
#     --output_dir /proj/berzelius-2023-364/users/x_qinzh/data/av2/h5py

# $PYTHON dataprocess/extract_av2.py --nproc 64 \
#     --av2_type sensor \
#     --data_mode val \
#     --argo_dir /proj/berzelius-2023-154/users/x_qinzh/av2 \
#     --output_dir /proj/berzelius-2023-364/users/x_qinzh/data/av2/h5py \
#     --mask_dir /proj/berzelius-2023-154/users/x_qinzh/av2/3d_scene_flow

# $PYTHON dataprocess/extract_av2.py --nproc 64 \
#     --av2_type sensor \
#     --data_mode test \
#     --argo_dir /proj/berzelius-2023-154/users/x_qinzh/av2 \
#     --output_dir /proj/berzelius-2023-364/users/x_qinzh/data/av2/h5py \
#     --mask_dir /proj/berzelius-2023-154/users/x_qinzh/av2/3d_scene_flow

# =============== Waymo Open Dataset
# $PYTHON dataprocess/extract_waymo.py --nproc 12 \
#     --map_dir /proj/berzelius-2023-154/users/x_qinzh/dataset/waymo/flowlabel/map \
#     --flow_data_dir /proj/berzelius-2023-154/users/x_qinzh/dataset/waymo/flowlabel \
#     --output_dir /proj/berzelius-2023-154/users/x_qinzh/dataset/waymo/h5py \
#     --mode train

$PYTHON dataprocess/extract_waymo.py --nproc 12 \
    --map_dir /proj/berzelius-2023-154/users/x_qinzh/dataset/waymo/flowlabel/map \
    --flow_data_dir /proj/berzelius-2023-154/users/x_qinzh/dataset/waymo/flowlabel \
    --output_dir /proj/berzelius-2023-154/users/x_qinzh/dataset/waymo/h5py \
    --mode valid

# =============== MAN-Truckscenes
# $PYTHON dataprocess/extract_truckscenes.py --nproc 12 \
#     --data_dir /proj/common-datasets/MAN-TruckScenes/v1.0 \
#     --mode v1.0-trainval \
#     --output_dir /proj/berzelius-2023-364/data/truckscenes/h5py \
#     --split_name val

# $PYTHON dataprocess/extract_truckscenes.py --nproc 12 \
#     --data_dir /proj/common-datasets/MAN-TruckScenes/v1.0 \
#     --mode v1.0-trainval \
#     --output_dir /proj/berzelius-2023-364/data/truckscenes/h5py \
#     --split_name train