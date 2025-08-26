"""
# Created: 2023-12-26 12:41
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: produce flow based on model predict and write into the dataset, 
#              then use `tools/visualization.py` to visualize the flow.
"""

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
import hydra, wandb, os, sys
from hydra.core.hydra_config import HydraConfig
from src.dataset import HDF5Dataset
from src.trainer import ModelWrapper
from src.utils import bc

@hydra.main(version_base=None, config_path="conf", config_name="save")
def main(cfg):
    pl.seed_everything(cfg.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    if 'iter_only' in cfg.model and cfg.model.iter_only:
        from src.runner import launch_runner
        print(f"---LOG[eval]: Run optmization-based method: {cfg.model.name}")
        cfg.res_name = cfg.model.name if (cfg.res_name is None) else cfg.res_name
        launch_runner(cfg, 'save')
        return
    
    if not os.path.exists(cfg.checkpoint):
        print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
        sys.exit(1)

    if cfg.res_name is None:
        cfg.res_name = cfg.checkpoint.split("/")[-1].split(".")[0]
        print(f"{bc.BOLD}NOTE{bc.ENDC}: res_name is not specified, use {bc.OKBLUE}{cfg.res_name}{bc.ENDC} as default.")

    checkpoint_params = DictConfig(torch.load(cfg.checkpoint)["hyper_parameters"])
    cfg.output = checkpoint_params.cfg.output
    cfg.model.update(checkpoint_params.cfg.model)
    mymodel = ModelWrapper.load_from_checkpoint(cfg.checkpoint, cfg=cfg, eval=True)

    wandb_logger = WandbLogger(save_dir=output_dir,
                               entity="kth-rpl",
                               project=f"deflow-eval", 
                               name=f"{cfg.output}",
                               offline=True)
    
    trainer = pl.Trainer(logger=wandb_logger, devices=1)
    # NOTE(Qingwen): search & check in pl_model.py : def test_step(self, batch, res_dict)
    trainer.test(model = mymodel, \
                 dataloaders = DataLoader(\
                     HDF5Dataset(cfg.dataset_path, n_frames=checkpoint_params.cfg.num_frames if 'num_frames' in checkpoint_params.cfg else 2), \
                    batch_size=1, shuffle=False))
    wandb.finish()

if __name__ == "__main__":
    main()