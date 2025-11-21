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

import os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import h5py
import numpy as np
from hydra.utils import instantiate
import shutil

from .dataset import HDF5Dataset
from .models.basic import cal_pose0to1
from .utils.eval_metric import OfficialMetrics, evaluate_leaderboard, evaluate_leaderboard_v2, evaluate_ssf
from .utils.av2_eval import write_output_file
from .utils.mics import zip_res
from .utils import InlineTee
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
            self.model = instantiate(cfg.model.target)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate model with target '{cfg.model.target}'. Error: {e}") from e
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
            is_eval_mode = False

        dataset = HDF5Dataset(dataset_path,
                              n_frames=self.cfg.num_frames,
                              eval=is_eval_mode,
                              leaderboard_version=self.cfg.leaderboard_version if 'leaderboard_version' in self.cfg else 1)

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

        batch['origin_pc0'] = batch['pc0'].clone()
        batch['pc0'] = batch['pc0'][~batch['gm0']].unsqueeze(0)
        batch['pc1'] = batch['pc1'][~batch['gm1']].unsqueeze(0)
        self.model.timer[12].start("One Scan")
        res_dict = self.model(batch)
        self.model.timer[12].stop()

        # NOTE (Qingwen): Since val and test, we will force set batch_size = 1 
        batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
        res_dict = {key: res_dict[key][0] for key in res_dict if (res_dict[key]!=None and len(res_dict[key])>0) }

        pc0 = batch['origin_pc0']
        pose_0to1 = cal_pose0to1(batch["pose0"], batch["pose1"])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0
        
        final_flow = pose_flow.clone()
        final_flow[~batch['gm0']] = res_dict['flow'] + pose_flow[~batch['gm0']]

        if self.mode in ['val', 'eval', 'valid']:
            eval_mask = batch['eval_mask'].squeeze()
            gt_flow = batch["flow"]
            v1_dict = evaluate_leaderboard(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                       gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], \
                                       batch['flow_category_indices'][eval_mask])
            v2_dict = evaluate_leaderboard_v2(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask],
                                                gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], 
                                                batch['flow_category_indices'][eval_mask])
            ssf_dict = evaluate_ssf(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                    gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], batch['flow_category_indices'][eval_mask])
            self.metrics.step(v1_dict, v2_dict, ssf_dict)

        elif self.mode == 'test':
            eval_mask = batch['eval_mask'].squeeze()
            save_pred_flow = final_flow[eval_mask, :3].cpu().detach().numpy()
            rigid_flow = pose_flow[eval_mask, :3].cpu().detach().numpy()
            is_dynamic = np.linalg.norm(save_pred_flow - rigid_flow, axis=1, ord=2) >= 0.05
            sweep_uuid = (batch['scene_id'], batch['timestamp'])

            if self.cfg.leaderboard_version == 2:
                save_pred_flow = (final_flow - pose_flow).cpu().detach().numpy()

            write_output_file(save_pred_flow, is_dynamic, sweep_uuid, self.save_res_path, self.cfg.leaderboard_version)
            
        elif self.mode == 'save':
            key = str(batch['timestamp'])
            scene_id = batch['scene_id']
            h5_path = os.path.join(self.cfg.dataset_path, f'{scene_id}.h5')
            try:
                with h5py.File(h5_path, 'r+') as f:
                    if self.cfg.res_name in f[key]:
                        del f[key][self.cfg.res_name]
                    f[key].create_dataset(self.cfg.res_name, data=final_flow.cpu().detach().numpy().astype(np.float32))
            except Exception as e:
                print(f"Rank {self.rank} failed to write to {h5_path}: {e}")

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

            with torch.no_grad():
                self._process_step(batch)
        
        iter_bar.close()

    def cleanup(self):
        dist.destroy_process_group()


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
                if metrics_obj is None: continue

                for key, val_list in metrics_obj.epe_3way.items():
                    final_metrics.epe_3way[key].extend(val_list)

                for class_idx, class_name in enumerate(metrics_obj.bucketedMatrix.class_names):
                    # NOTE(Qingwen): for bucketedMatrix range_buckets = speed_buckets
                    for speed_idx, speed_bucket in enumerate(metrics_obj.bucketedMatrix.range_buckets):
                        count = metrics_obj.bucketedMatrix.count_storage_matrix[class_idx, speed_idx]
                        if count > 0:
                            avg_epe = metrics_obj.bucketedMatrix.epe_storage_matrix[class_idx, speed_idx]
                            avg_speed = metrics_obj.bucketedMatrix.range_storage_matrix[class_idx, speed_idx]
                            final_metrics.bucketedMatrix.accumulate_value(
                                class_name, speed_bucket, avg_epe, avg_speed, count
                            )
                for class_idx, class_name in enumerate(metrics_obj.distanceMatrix.class_names):
                    # NOTE(Qingwen): for distanceMatrix range_buckets = distance_buckets
                    for range_idx, range_bucket in enumerate(metrics_obj.distanceMatrix.range_buckets):
                        count = metrics_obj.distanceMatrix.count_storage_matrix[class_idx, range_idx]
                        if count > 0:
                            avg_epe = metrics_obj.distanceMatrix.epe_storage_matrix[class_idx, range_idx]
                            avg_range = metrics_obj.distanceMatrix.range_storage_matrix[class_idx, range_idx]
                            final_metrics.distanceMatrix.accumulate_value(
                                class_name, range_bucket, avg_epe, avg_range, count
                            )

            final_metrics.print() 
        else:
            print(f"\nWe already write the {cfg.res_name} into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:")
            print(f"python tools/visualization.py --res_name '{cfg.res_name}' --data_dir {cfg.dataset_path}")
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
        print(f"---LOG[eval]: Run optimization-based method: {cfg.model.name} on {cfg.dataset_path}/{cfg.data_mode} set.\n")
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
            final_zip_file = zip_res(
                cfg.save_res_path, 
                leaderboard_version=cfg.leaderboard_version, 
                is_supervised=False,  # all optimization-based now is ssl.
                output_file=str(cfg.save_res_path) + ".zip"
            )
            print(f"--- [LOG] Submission file successfully created at: {final_zip_file}")
        return
    else:
        _run_process(cfg, mode)
