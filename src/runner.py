"""
# Created: 2025-08-09 18:59
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), AI Assistant (Google AI Studio)
#
# This file is part of 
# * SeFlow (https://github.com/KTH-RPL/SeFlow) 
# * HiMo (https://kin-zhang.github.io/HiMo)
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
#
# Description: Runner for optimization-based models (e.g., NSFP, FastNSF) that
#              replaces the need for PyTorch Lightning in evaluation/testing.
#              It handles multi-GPU distribution, execution, and result aggregation.
# 
"""

import os, sys, random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import open_dict
import shutil

from .dataset import HDF5Dataset
from .utils.eval_metric import OfficialMetrics
from .utils.mics import zip_res, enable_timer_sync
from .utils import InlineTee
from .postprocess import run_model_wo_ground, flow_on_all_points, eval_frame, write_flow_to_h5, write_leaderboard_file

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # cpu and all gpus

def load_model(cfg):
    """Optimization-based methods (iter_only) have no weights, feed-forward ones load them from cfg.checkpoint."""
    if cfg.model.get('iter_only', False):
        return instantiate(cfg.model.target)

    # NOTE: same setting as src/trainer.py, to have the same scores as before on the released checkpoints.
    torch.set_float32_matmul_precision('medium')
    target = cfg.model.target
    with open_dict(target):
        target['grid_feature_size'] = [abs(int((target.point_cloud_range[i] - target.point_cloud_range[i+3]) / target.voxel_size[i])) for i in range(3)]
    model = instantiate(target)
    # the checkpoint is saved by lightning in train.py, the model weights are under 'model.'
    # weights_only=False: it also has the training config (omegaconf objects), which torch>=2.6 refuses to load by default.
    state_dict = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")})
    return model.eval()

class SceneDistributedSampler(Sampler):
    """
    A DistributedSampler that distributes data based on scene IDs, not individual indices.
    This ensures that all frames from a single scene are processed by the same GPU,
    preventing I/O conflicts when writing to HDF5 files.

    Args:
        dataset (Dataset): The dataset to sample from. It must have a `data_index`
                           attribute where each element is a list/tuple like [scene_id, timestamp].
        num_replicas (int, optional): Number of processes participating in distributed training.
                                      Defaults to world_size.
        rank (int, optional): Rank of the current process. Defaults to current rank.
    """
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

        # 1. Group indices by scene_id
        scene_to_indices = defaultdict(list)
        data_index = self.dataset.eval_data_index if self.dataset.eval_index else self.dataset.data_index
        for idx, (scene_id, _) in enumerate(data_index):
            scene_to_indices[scene_id].append(idx)
        
        self.scene_to_indices = scene_to_indices
        self.scenes = sorted(list(self.scene_to_indices.keys()))

        # 2. Distribute the list of scenes among replicas
        scenes_for_this_rank = self.scenes[self.rank:len(self.scenes):self.num_replicas]

        # 3. Flatten the indices for the assigned scenes
        self.indices = []
        for scene_id in scenes_for_this_rank:
            self.indices.extend(self.scene_to_indices[scene_id])
            
        self.num_samples = len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples
    
class InferenceRunner:
    def __init__(self, cfg, rank, world_size, mode):
        try:
            self.model = load_model(cfg)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model with target '{cfg.model.target}'. Error: {e}") from e
        if cfg.get('timer_sync', False):
            enable_timer_sync(self.model)
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{self.rank}")
        # 'val' for validation, 'test' for test submission, 'save' for h5 writing
        self.mode = mode

        self.model.to(self.device)
        self.metrics = OfficialMetrics() if self.mode in ['val', 'eval', 'valid'] else None
        self.res_name = cfg.get('res_name', cfg.model.name)
        self.save_res_path = cfg.get('save_res_path', None)

    def _setup_dataloader(self):
        if self.mode in ['val', 'test', 'eval', 'valid']:
            dataset_path = self.cfg.dataset_path + f"/{self.cfg.data_mode}"
            is_eval_mode = True
        else: # 'save'
            dataset_path = self.cfg.dataset_path
            is_eval_mode = self.cfg.get('val_index_only', False)

        dataset = HDF5Dataset(dataset_path,
                              n_frames=self.cfg.num_frames,
                              future_frames=self.cfg.get('future_frames', 1),
                              pad_frames=self.cfg.model.get('pad_frames', True),
                              eval=is_eval_mode)

        sampler = SceneDistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
        
        return DataLoader(dataset, 
                          batch_size=1, # One sample per optimization
                          sampler=sampler, 
                          num_workers=self.cfg.get('num_workers', 4),
                          pin_memory=True)

    
    def _process_step(self, batch):
        # Move all tensors to the appropriate device
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(self.device)
            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], torch.Tensor):
                batch[key] = [item.to(self.device) for item in val]

        batch, res_dict = run_model_wo_ground(self.model, batch)
        final_flow, pose_flow = flow_on_all_points(batch, res_dict)

        if self.mode in ['val', 'eval', 'valid']:
            self.metrics.step(*eval_frame(batch, final_flow, pose_flow))
            if self.cfg.get('save_res', False):
                data_dir = os.path.join(self.cfg.dataset_path, self.cfg.data_mode)
                write_flow_to_h5(data_dir, batch['scene_id'], batch['timestamp'], self.res_name, final_flow)
        elif self.mode == 'test':
            write_leaderboard_file(batch, final_flow, pose_flow, self.save_res_path)
        elif self.mode == 'save':
            write_flow_to_h5(self.cfg.dataset_path, batch['scene_id'], batch['timestamp'], self.res_name, final_flow)

    def run(self):
        dataloader = self._setup_dataloader()
        

        iter_bar = tqdm(
            dataloader, 
            desc=f"[GPU {self.rank}] Initializing...", 
            position=self.rank,
            leave=False,
            dynamic_ncols=True
        )

        for batch in iter_bar:
            current_scene_id = batch['scene_id'][0] 
            
            iter_bar.set_description(f"[GPU {self.rank}] Processing {current_scene_id}")
            # same random state for every frame: the result of a frame doesn't depend on the frames before it or on how many GPUs we use.
            seed_everything(self.cfg.seed)

            with torch.no_grad():
                self._process_step(batch)
        
        iter_bar.close()

    def cleanup(self):
        dist.destroy_process_group()
        # worker processes that a library left behind (joblib inside hdbscan, for ICP-Flow) keep this process alive
        # until they time out (300s with joblib>=1.3), so stop them here.
        for child in mp.active_children():
            child.terminate()


def _setup_distributed_environment():
    """
    Sets up the distributed environment. Prioritizes Slurm variables if available,
    otherwise assumes torchrun/mp.spawn and initializes based on environment variables.
    """
    if 'SLURM_PROCID' in os.environ:
        # --- Slurm Environment ---
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        hostnames = os.environ['SLURM_JOB_NODELIST']
        master_addr = hostnames.split(',')[0].split('(')[0]
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    # Let torch.distributed handle the rest, it will use the environment variables
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size()


def _run_process(cfg, mode):
    rank, world_size = _setup_distributed_environment()
    
    runner = InferenceRunner(cfg, rank, world_size, mode)
    runner.run()

    # --- use dist.gather_object to synchronize metrics ---
    if world_size > 1:
        gathered_metrics_objects = [None] * world_size if rank == 0 else None
        
        dist.gather_object(
            runner.metrics,
            gathered_metrics_objects if rank == 0 else None,
            dst=0
        )
    else:
        gathered_metrics_objects = [runner.metrics]

    if rank == 0:
        if mode in ['val', 'eval', 'valid']:
            final_metrics = OfficialMetrics()
            print(f"\n--- [LOG] Finished processing. Aggregating results from {world_size} GPUs with {len(gathered_metrics_objects)} metrics objects...")
            for metrics_obj in gathered_metrics_objects:
                if metrics_obj is not None:
                    final_metrics.merge(metrics_obj)

            final_metrics.print()
        if mode == 'save' or (cfg.get('save_res', False) and mode != 'test'):
            data_dir = cfg.dataset_path if mode == 'save' else f"{cfg.dataset_path}/{cfg.data_mode}"
            print(f"\nWe already write the {runner.res_name} into {data_dir}/layers/pred/{runner.res_name}, please run following commend to visualize the flow. Copy and paste it to your terminal:")
            print(f"python tools/visualization.py --res_name '{runner.res_name}' --data_dir {data_dir}")
            print(f"Enjoy! ^v^ ------ \n")
        runner.model.timer.print(random_colors=False, bold=False)

    if world_size > 1:
        dist.barrier()
        
    runner.cleanup()

def _spawn_wrapper(rank, world_size, cfg, mode, output_dir):
    log_filepath = f"{output_dir}/output.log" if output_dir else None
    if log_filepath and rank==0:
        sys.stdout = InlineTee(log_filepath, append=True)
    if rank == 0:
        data_path = cfg.dataset_path if mode == 'save' else f"{cfg.dataset_path}/{cfg.data_mode}" # save.py has no data_mode
        if cfg.model.get('iter_only', False):
            print(f"---LOG[eval]: Run optimization-based method: {cfg.model.name} on {data_path} set.\n")
        else:
            print(f"---LOG[eval]: Run {cfg.model.name} with checkpoint {cfg.checkpoint} on {data_path} set.\n")
    torch.cuda.set_device(rank)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(cfg.get('master_port', 12355))
    _run_process(cfg, mode)

def launch_runner(cfg, mode, output_dir):
    is_slurm_job = 'SLURM_PROCID' in os.environ
    
    if not is_slurm_job and not dist.is_initialized():
        world_size = torch.cuda.device_count()
        if world_size == 0:
            raise SystemError("No CUDA devices found.")
        
        if mode == 'test':
            cfg.save_res_path = Path(cfg.dataset_path).parent / "results" / cfg.output
            
        mp.spawn(_spawn_wrapper,
                 args=(world_size, cfg, mode, output_dir),
                 nprocs=world_size,
                 join=True)
        
        if mode == 'test':
            print("\nAll workers finished. Aggregating results into submission file...")
            final_zip_file = zip_res(cfg.save_res_path, output_file=str(cfg.save_res_path) + ".zip")
            print(f"--- [LOG] Submission file successfully created at: {final_zip_file}")
            print(f"""
curl -O https://sceneflow.kth-rpl.se/download/sceneflow_submit.sh
export SCENEFLOW_API_KEY=your_api_key
bash sceneflow_submit.sh {cfg.save_res_path}.zip --method-name {cfg.save_res_path.name} --wait
""")
        return
    else:
        _run_process(cfg, mode)
