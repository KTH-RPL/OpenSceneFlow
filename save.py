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
from omegaconf import DictConfig
import hydra, os, sys
from hydra.core.hydra_config import HydraConfig
from src.runner import launch_runner
from src.utils import bc

@hydra.main(version_base=None, config_path="conf", config_name="save")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir

    if cfg.model.get('iter_only', False):
        print(f"---LOG[eval]: Run optmization-based method: {cfg.model.name}")
        cfg.res_name = cfg.model.name if (cfg.res_name is None) else cfg.res_name
    else:
        # feed-forward model: its model config is inside the checkpoint.
        if not os.path.exists(cfg.checkpoint):
            print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
            sys.exit(1)

        if cfg.res_name is None:
            cfg.res_name = cfg.checkpoint.split("/")[-1].split(".")[0]
            print(f"{bc.BOLD}NOTE{bc.ENDC}: res_name is not specified, use {bc.OKBLUE}{cfg.res_name}{bc.ENDC} as default.")

        checkpoint_params = DictConfig(torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)["hyper_parameters"])
        cfg.output = checkpoint_params.cfg.output
        cfg.model.update(checkpoint_params.cfg.model)
        cfg.num_frames = cfg.model.target.get('num_frames', checkpoint_params.cfg.get('num_frames', cfg.get('num_frames', 2)))

    # NOTE(Qingwen): search & check: def _process_step(self, batch) in src/runner.py
    launch_runner(cfg, 'save', output_dir)

if __name__ == "__main__":
    main()
