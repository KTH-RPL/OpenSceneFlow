"""
# Created: 2023-08-09 10:28
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: Output the evaluation results, go for local evaluation or online evaluation
"""

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig
import hydra, wandb, os, sys
from hydra.core.hydra_config import HydraConfig
from src.dataset import HDF5Dataset
from src.trainer import ModelWrapper
from src.utils import InlineTee

def precheck_cfg_valid(cfg):
    if os.path.exists(cfg.dataset_path + f"/{cfg.data_mode}") is False:
        raise ValueError(f"Dataset {cfg.dataset_path}/{cfg.data_mode} does not exist. Please check the path.")
    if cfg.supervised_flag not in [True, False]:
        raise ValueError(f"Supervised flag {cfg.supervised_flag} is not valid. Please set it to True or False.")
    if cfg.leaderboard_version not in [1, 2]:
        raise ValueError(f"Leaderboard version {cfg.leaderboard_version} is not valid. Please set it to 1 or 2.")
    return cfg

@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    if 'iter_only' in cfg.model and cfg.model.iter_only:
        from src.runner import launch_runner
        launch_runner(cfg, cfg.data_mode, output_dir)
        print(f"---LOG[eval]: Finished optimization-based evaluation. Logging saved to {output_dir}/output.log")
        return
    
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)
        
    torch_load_ckpt = torch.load(cfg.checkpoint)
    checkpoint_params = DictConfig(torch_load_ckpt["hyper_parameters"])
    cfg.output = checkpoint_params.cfg.output + f"-e{torch_load_ckpt['epoch']}-{cfg.data_mode}-v{cfg.leaderboard_version}"
    # replace output_dir ${old_output_dir} with ${output_dir}
    output_dir = output_dir.replace(HydraConfig.get().runtime.output_dir.split('/')[-2], checkpoint_params.cfg.output.split('/')[-1])
    cfg.model.update(checkpoint_params.cfg.model)
    cfg.num_frames = cfg.model.target.get('num_frames', checkpoint_params.cfg.get('num_frames', cfg.get('num_frames', 2)))
    
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)
    os.makedirs(output_dir, exist_ok=True)
    sys.stdout = InlineTee(f"{output_dir}/output.log")
    print(f"---LOG[eval]: Loaded model from {cfg.checkpoint}. The backbone network is {checkpoint_params.cfg.model.name}.")
    print(f"---LOG[eval]: Evaluation data: {cfg.dataset_path}/{cfg.data_mode} set.\n")

    if cfg.wandb_mode != "disabled":
        logger = WandbLogger(save_dir=output_dir,
                            entity="kth-rpl",
                            project=f"opensf-eval", 
                            name=f"{cfg.output}",
                            offline=(cfg.wandb_mode == "offline"))
        logger.watch(mymodel, log_graph=False)
    else:
        # check local tensorboard logging: tensorboard --logdir logs/jobs/{log folder}
        logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    
    trainer = pl.Trainer(logger=logger, devices=1)
    # NOTE(Qingwen): search & check: def eval_only_step_(self, batch, res_dict)
    trainer.validate(model = mymodel, \
                     dataloaders = DataLoader( \
                                            HDF5Dataset(cfg.dataset_path + f"/{cfg.data_mode}", \
                                                        n_frames=cfg.num_frames, \
                                                        eval=True, leaderboard_version=cfg.leaderboard_version), \
                                            batch_size=1, shuffle=False))
    if cfg.wandb_mode != "disabled":
        wandb.finish()
    print(f"---LOG[eval]: Finished feed-forward evaluation. Logging saved to {output_dir}/output.log")

if __name__ == "__main__":
    main()