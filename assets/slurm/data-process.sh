#!/bin/bash
#SBATCH -J dataprocess
#SBATCH -A berzelius-2026-200
#SBATCH -p berzelius-cpu
#SBATCH --cpus-per-task 32
#SBATCH --mem 128G
#SBATCH -t 2-00:00:00
#SBATCH --output /proj/berzelius-2023-364/users/x_qinzh/workspace/OpenSceneFlow/logs/slurm/%j_data.out
#SBATCH --error  /proj/berzelius-2023-364/users/x_qinzh/workspace/OpenSceneFlow/logs/slurm/%j_data.err

# Usage: sbatch assets/slurm/data-process.sh {nus|man|av2} {train|val}  or  waymo {training|validation}
DATASET=$1
SPLIT=$2

PYTHON=/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/sftool/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib
cd /proj/berzelius-2023-364/users/x_qinzh/workspace/OpenSceneFlow/
OUT=/proj/berzelius-2023-364/data/opensf-h5py-v2
NPROC=$((SLURM_CPUS_PER_TASK - 2))
echo "Job ${SLURM_JOB_ID}: ${DATASET} ${SPLIT} with ${NPROC} processes"

case ${DATASET} in
  nus)
    $PYTHON dataprocess/extract_nus.py --nproc ${NPROC} \
        --data_dir /proj/berzelius-2023-364/users/x_qinzh/data/nus/raw \
        --mode v1.0-trainval --split_name ${SPLIT} \
        --output_dir ${OUT}/nus
    ;;
  man)
    $PYTHON dataprocess/extract_truckscenes.py --nproc ${NPROC} \
        --data_dir /proj/common-datasets/MAN-TruckScenes/v1.0 \
        --mode v1.0-trainval --split_name ${SPLIT} \
        --output_dir ${OUT}/man
    ;;
  av2)
    # argo_dir/sensor -> /proj/common-datasets/Argoverse/v2.0/Sensor
    $PYTHON dataprocess/extract_av2.py --nproc ${NPROC} \
        --av2_type sensor --data_mode ${SPLIT} \
        --argo_dir /proj/berzelius-2023-364/users/x_qinzh/workspace/av2-raw \
        --mask_dir /proj/berzelius-2023-364/users/x_qinzh/workspace/av2-raw/3d_scene_flow \
        --output_dir ${OUT}/av2
    ;;
  waymo)
    # Waymo v2.0.1 parquet, SPLIT: training | validation, needs the waymo2 env (py3.10 + waymo-open-dataset-tf-2-12-0==1.6.4)
    /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/waymo2/bin/python dataprocess/extract_waymo_v2.py --nproc ${NPROC} \
        --data_dir /proj/common-datasets/Waymo/Perception/2.0.1 \
        --mode ${SPLIT} \
        --output_dir ${OUT}/waymo
    [ "${SPLIT}" = "validation" ] && $PYTHON tools/create_evalpkl.py --data_dir ${OUT}/waymo/valid
    ;;
  *)
    echo "unknown dataset ${DATASET}"; exit 1 ;;
esac

# fail the job (so `--dependency=afterok` jobs don't start) when nothing was extracted, e.g. a wrong input path
SPLIT_DIR=$(case ${DATASET} in av2) echo ${OUT}/av2/sensor/${SPLIT};; waymo) echo ${OUT}/waymo/$([ "${SPLIT}" = "training" ] && echo train || echo valid);; *) echo ${OUT}/${DATASET}/${SPLIT};; esac)
NUM_H5=$(ls ${SPLIT_DIR}/*.h5 2>/dev/null | wc -l)
echo "${NUM_H5} scenes in ${SPLIT_DIR}"
[ ${NUM_H5} -gt 0 ] || exit 1
exit 0
