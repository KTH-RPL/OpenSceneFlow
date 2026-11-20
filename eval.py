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
from omegaconf import DictConfig
import hydra, os, sys
from hydra.core.hydra_config import HydraConfig
from src.runner import launch_runner

@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    if cfg.data_mode == 'test' and cfg.leaderboard_version != 3:
        raise ValueError("The v1 and v2 leaderboards are closed, only leaderboard_version=3 (https://sceneflow.kth-rpl.se/leaderboard) can be submitted.")

    # feed-forward model: its model config is inside the checkpoint. Optimization-based one (iter_only) has no checkpoint.
    if not cfg.model.get('iter_only', False):
        if not os.path.exists(cfg.checkpoint):
            print(f"Checkpoint {cfg.checkpoint} does not exist. Need checkpoints for evaluation.")
            sys.exit(1)

        torch_load_ckpt = torch.load(cfg.checkpoint, map_location="cpu")
        checkpoint_params = DictConfig(torch_load_ckpt["hyper_parameters"])
        cfg.output = checkpoint_params.cfg.output + f"-e{torch_load_ckpt['epoch']}-{cfg.data_mode}-v{cfg.leaderboard_version}"
        # replace output_dir ${old_output_dir} with ${output_dir}
        output_dir = output_dir.replace(HydraConfig.get().runtime.output_dir.split('/')[-2], checkpoint_params.cfg.output.split('/')[-1])
        cfg.model.update(checkpoint_params.cfg.model)
        cfg.num_frames = cfg.model.target.get('num_frames', checkpoint_params.cfg.get('num_frames', cfg.get('num_frames', 2)))
        os.makedirs(output_dir, exist_ok=True)

    # NOTE(Qingwen): search & check: def _process_step(self, batch) in src/runner.py
    launch_runner(cfg, cfg.data_mode, output_dir)
    print(f"---LOG[eval]: Finished evaluation. Logging saved to {output_dir}/output.log")

if __name__ == "__main__":
    main()
