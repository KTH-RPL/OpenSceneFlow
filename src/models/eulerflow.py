"""
# Created: 2026-11-21 09:30
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), AI Assistant (Claude Code)
#
# This file is part of OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow).
# If you find this repo helpful, please cite the respective publication as
# listed on the above website.
#
# Description: Re-implementation of EulerFlow (ICLR 2025): Neural Eulerian Scene Flow Fields.
#              One MLP is optimized for a whole scene: [x, y, z, time, direction] -> flow to the next / previous frame.
#              Follows the released code https://github.com/kylevedder/SceneFlowZoo (MIT license, Copyright (c) Kyle Vedder),
#              models/mini_batch_optimization/eulerflow.py with configs/eulerflow/argo/noncausal_minibatched/val.py,
#              written here in one file, and with our CUDA chamfer for the nearest neighbors.
#
# Different from the other methods: the runner calls fit_scene() once with all frames of a scene, then forward() for every frame.
# The fit of one scene is saved in ckpt_dir and a run that was stopped goes on from there.
#
# The fit of SceneFlowZoo (1000 epochs on all points, learning rate 8e-5) takes hours per scene. `schedule` makes it shorter:
# first a high learning rate on random points of every frame, then a low learning rate. See conf/model/eulerflow.yaml.
#
# NOTE: SceneFlowZoo only optimizes (and evaluates) the first 150 frames of a scene, here it is every frame of the scene.
"""

import os, time, copy
import dztimer, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .basic import cal_pose0to1, in_range
from .basic.opt_module import EarlyStopping
from assets.cuda.chamfer3D import nnChamferDis
MyCUDAChamferDis = nnChamferDis()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def time_mlp(num_layers, width):
    """[x, y, z, time, direction] -> flow. The layers of the NSFP network (Neural_Prior), with two more inputs."""
    layers = [nn.Linear(5, width), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers += [nn.Linear(width, width), nn.ReLU()]
    layers.append(nn.Linear(width, 3))
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.0)
    return nn.Sequential(*layers)


def query_flow(net, pc, time, direction):
    """Flow of the points pc [N, 3] at `time`, to the next frame (direction 1) or to the previous one (direction -1)."""
    time_direction = pc.new_tensor([time, direction]).expand(pc.shape[0], 2)
    return net(torch.cat([pc, time_direction], dim=1))


class EulerFlow(nn.Module):
    def __init__(self, num_layers=8, width=128, window=5, k_step=3, cycle_weight=1.0, speed_threshold=6.0,
                 epochs=1000, schedule=[[0, 8e-5, 0]], early_patience=150, min_delta=0.001,
                 point_cloud_range=[-48, -48, -2.5, 48, 48, 2.5], ckpt_dir='logs/eulerflow', save_every=10):
        super().__init__()
        # what the fit of a scene depends on. A saved fit with other values is not used.
        self.hparams = dict(num_layers=num_layers, width=width, window=window, k_step=k_step, cycle_weight=cycle_weight,
                            speed_threshold=speed_threshold, schedule=[[int(start), float(lr), int(points)] for start, lr, points in schedule],
                            point_cloud_range=[float(v) for v in point_cloud_range])
        self.num_layers, self.width = num_layers, width
        self.window = window # number of frames in one optimization step
        self.k_step = k_step # the points are also moved k frames along the flow (Euler integration) and compared with that frame. 1: off.
        self.cycle_weight = cycle_weight
        self.speed_threshold = speed_threshold # meter per frame, a flow longer than this is penalized
        self.epochs = epochs
        # stages of the fit, [[from epoch, learning rate, points per frame], ...], the first one from epoch 0.
        # points per frame: 0: every step moves all points of a frame. N: N random points of it, faster and less memory.
        self.schedule = self.hparams['schedule']
        self.early_patience, self.min_delta = early_patience, min_delta
        self.save_every = save_every # epochs between two saves of a running fit
        self.ckpt_dir = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(BASE_DIR, ckpt_dir)
        self.register_buffer('pc_range', torch.tensor(self.hparams['point_cloud_range']))

        self.net = time_mlp(num_layers, width)
        self.to_global, self.time = {}, {} # of every frame of the fitted scene, by timestamp

        self.timer = dztimer.Timing()
        self.timer.start("EulerFlow Model Inference")
        print(f"\n---LOG [model]: EulerFlow setup num_layers: {num_layers}, window: {window}, k_step: {k_step}, epochs: {epochs}, schedule: {self.schedule}, fits saved in {self.ckpt_dir}.")

    def window_loss(self, net, pcs, targets, times):
        """
        The loss of one optimization step, on `window` frames next to each other.
        pcs: the points of these frames that are moved along the flow, in the same coordinate (all points, or the random ones of this step).
        targets: all points of these frames, the moved points are compared with them. times: the time of these frames.
        """
        loss = 0
        one_step = {} # (frame, direction): the points of the frame moved one step, and their distance to the frame they arrive at
        for direction in (1, -1): # along the time, and against it
            # one step, from every frame that has a neighbor in this direction
            for i in (range(len(pcs) - 1) if direction == 1 else range(1, len(pcs))):
                flow = query_flow(net, pcs[i], times[i], direction)
                pc = pcs[i] + flow
                # squared distance to the nearest point, both ways. A distance >= 2.0 counts as 0.
                distance = MyCUDAChamferDis.truncated_dis(pc, targets[i + direction], truncate_dist=2.0)
                too_fast = F.relu(flow.norm(dim=1) - self.speed_threshold)
                loss = loss + distance + too_fast.mean() + 0.5 * (too_fast ** 2).mean()
                one_step[i, direction] = (pc, distance)

            # k steps along the flow (Euler integration), from every frame that has k neighbors in this direction
            if self.k_step > 1:
                for i in (range(len(pcs) - self.k_step) if direction == 1 else range(self.k_step, len(pcs))):
                    pc, distance = one_step[i, direction]
                    loss = loss + distance # the first step is the one above (SceneFlowZoo computes it a second time)
                    for step in range(2, self.k_step + 1):
                        # NOTE: SceneFlowZoo asks the flow at the time of the start frame in every step (not times[i + direction * (step - 1)]),
                        # we keep it to have the same loss as the released code.
                        pc = pc + query_flow(net, pc, times[i], direction)
                        loss = loss + MyCUDAChamferDis.truncated_dis(pc, targets[i + direction * step], truncate_dist=2.0)

        # cycle consistency: a point that goes to the next frame and comes back should be where it started
        for i in range(len(pcs) - 1):
            pc_next = one_step[i, 1][0]
            pc_back = pc_next + query_flow(net, pc_next, times[i + 1], -1)
            # NOTE: SceneFlowZoo writes `* 0.01` for this term, but its cost classes drop the factor when they sum
            # (checked by running them), so the released results have the weight 1.0.
            loss = loss + self.cycle_weight * torch.sigmoid((pc_back - pcs[i]).norm(dim=1)).mean()
        return loss

    def fit_scene(self, scene_id, frames):
        """
        Optimize the network on one scene. The runner calls it once, before the first frame of the scene.
        frames: every frame of the scene in time order, [{'timestamp', 'pc': [N, 3], 'gm': [N] ground mask, 'pose': [4, 4]}, ...] as numpy.
        """
        device = self.pc_range.device
        # All frames go into one coordinate: the one of the middle frame, as SceneFlowZoo (the network takes the raw x, y, z).
        origin_pose = frames[(len(frames) - 1) // 2 + 1]['pose']
        self.to_global, self.time, pcs = {}, {}, []
        for i, frame in enumerate(frames):
            to_global = torch.from_numpy(np.linalg.inv(origin_pose.astype(np.float64)) @ frame['pose'].astype(np.float64)).float().to(device)
            self.to_global[str(frame['timestamp'])] = to_global
            self.time[str(frame['timestamp'])] = i / (len(frames) - 1) - 0.5
            pc = torch.from_numpy(frame['pc'][~frame['gm']]).float().to(device)
            pcs.append(pc[in_range(pc, self.pc_range)] @ to_global[:3, :3].T + to_global[:3, 3])
        times = list(self.time.values())

        net = time_mlp(self.num_layers, self.width).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.schedule[0][1], weight_decay=0)
        early_stopping = EarlyStopping(patience=self.early_patience, min_delta=self.min_delta)
        state = {'epoch': 0, 'best_loss': torch.inf, 'best_net': None, 'early_stopped': False, 'hparams': self.hparams}

        ckpt_path = os.path.join(self.ckpt_dir, f'{scene_id}.pth')
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            if state['hparams'] != self.hparams:
                raise ValueError(f"{ckpt_path} was fitted with other parameters: {state['hparams']}, now: {self.hparams}. "
                                 f"Remove it or set another model.target.ckpt_dir.")
            net.load_state_dict(state['net'])
            optimizer.load_state_dict(state['optimizer'])
            early_stopping.best, early_stopping.num_bad_epochs = state['early_stopping']
            print(f"---LOG [EulerFlow]: {scene_id} is read from {ckpt_path} (epoch {state['epoch']}{', early stopped' if state['early_stopped'] else ''}).")

        def save():
            state.update(net=net.state_dict(), optimizer=optimizer.state_dict(), early_stopping=(early_stopping.best, early_stopping.num_bad_epochs))
            os.makedirs(self.ckpt_dir, exist_ok=True)
            torch.save(state, ckpt_path + '.tmp')
            os.replace(ckpt_path + '.tmp', ckpt_path) # a run that is killed while saving leaves the last file as it was

        num_windows = len(pcs) - self.window + 1
        start_time = time.time()
        with torch.inference_mode(False), torch.enable_grad():
            while state['epoch'] < self.epochs and not state['early_stopped']:
                # the stage of this epoch: the last one of the schedule that has started
                _, lr, points = [stage for stage in self.schedule if stage[0] <= state['epoch']][-1]
                for group in optimizer.param_groups:
                    group['lr'] = lr
                # one epoch: one step on every window of the scene, in random order (same order as SceneFlowZoo)
                epoch_loss = 0
                random_points = torch.Generator(device=device).manual_seed(state['epoch']) # seeded by the epoch as the order, a continued fit takes the same points
                for start in np.random.default_rng(state['epoch']).permutation(num_windows):
                    targets = pcs[start:start + self.window]
                    moved = targets if points == 0 else [pc[torch.randperm(len(pc), device=device, generator=random_points)[:points]] for pc in targets]
                    optimizer.zero_grad()
                    loss = self.window_loss(net, moved, targets, times[start:start + self.window])
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                state['epoch'] += 1
                if epoch_loss < state['best_loss']:
                    state['best_loss'], state['best_net'] = epoch_loss, copy.deepcopy(net.state_dict())
                state['early_stopped'] = early_stopping.step(torch.tensor(epoch_loss / num_windows))
                if state['early_stopped'] or state['epoch'] == self.epochs or state['epoch'] % self.save_every == 0:
                    save()
                    print(f"---LOG [EulerFlow]: {scene_id} epoch {state['epoch']}/{self.epochs}, loss {epoch_loss / num_windows:.4f} "
                          f"(best {state['best_loss'] / num_windows:.4f}), {(time.time() - start_time) / 60:.1f} min in this run.", flush=True)

        self.net.load_state_dict(state['best_net'])

    def forward(self, batch):
        batch_flow = []
        for batch_id in range(len(batch["pose0"])):
            timestamp = batch['timestamp'][batch_id]
            timestamp = str(timestamp.item() if torch.is_tensor(timestamp) else timestamp)
            if timestamp not in self.to_global:
                raise RuntimeError(f"Frame {timestamp} is not in the scene EulerFlow is fitted on, fit_scene() has to be called first.")
            pc0 = batch["pc0"][batch_id]
            pc0_in_range = in_range(pc0, self.pc_range)
            to_global = self.to_global[timestamp]
            global_flow = query_flow(self.net, pc0[pc0_in_range] @ to_global[:3, :3].T + to_global[:3, 3], self.time[timestamp], 1)

            # The runner adds the ego motion itself: it wants the rest of the flow, in the coordinate of pc1.
            pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id])
            global_to_pc1 = pose_0to1[:3, :3] @ to_global[:3, :3].T
            flow = torch.zeros_like(pc0)
            flow[pc0_in_range] = global_flow @ global_to_pc1.T
            batch_flow.append(flow)

        return {"flow": batch_flow}
