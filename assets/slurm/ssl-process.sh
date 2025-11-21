#!/bin/bash
#SBATCH -J ssl-label
#SBATCH -p berzelius-cpu
#SBATCH --cpus-per-task 64
#SBATCH --mem 128G
#SBATCH --mincpus=64
#SBATCH -t 2-00:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qingwen@kth.se
#SBATCH --output /proj/berzelius-2023-154/users/x_qinzh/OpenSceneFlow/logs/slurm/data/%A-%a.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_qinzh/OpenSceneFlow/logs/slurm/data/%A-%a.err


PYTHON=/proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/sftool/bin/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib
cd /proj/berzelius-2023-154/users/x_qinzh/OpenSceneFlow


# data directory containing the extracted h5py files
DATA_DIR="/proj/berzelius-2023-364/data/av2/h5py/sensor/train"

TOTAL_SCENES=$(ls ${DATA_DIR}/*.h5 | wc -l)
# Process every n-th frame into DUFOMap, no need to change at least for now.
INTERVAL=1
NUM_TOTAL_ITEMS=$((TOTAL_SCENES + 1))
SPLIT_SIZE=$(( (NUM_TOTAL_ITEMS + SLURM_ARRAY_TASK_COUNT - 1) / SLURM_ARRAY_TASK_COUNT ))
echo "===================================================="
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total jobs in array: ${SLURM_ARRAY_TASK_COUNT}"
echo "Total scenes to process: $((TOTAL_SCENES + 1)) (from 0 to ${TOTAL_SCENES})"

# --- Task Logic ---
# SLURM_ARRAY_TASK_ID is the index of the current job in the array (e.g., 0, 1, 2,...)
START_SCENE=$((SLURM_ARRAY_TASK_ID * SPLIT_SIZE))

END_SCENE=$((START_SCENE + SPLIT_SIZE))
if [ $END_SCENE -gt $((TOTAL_SCENES + 1)) ]; then
    END_SCENE=$((TOTAL_SCENES + 1))
fi

echo "Running job for task id ${SLURM_ARRAY_TASK_ID}"
echo "Processing scene range: ${START_SCENE} to ${END_SCENE}"
$PYTHON process.py --data_dir ${DATA_DIR} --interval ${INTERVAL} --scene_range ${START_SCENE},${END_SCENE}
