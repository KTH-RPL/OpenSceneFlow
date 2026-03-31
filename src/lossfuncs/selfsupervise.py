"""
# Created: 2023-07-17 00:00
# Updated: 2025-08-07 00:01
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of 
# * SeFlow (https://github.com/KTH-RPL/SeFlow)
# * HiMo (https://kin-zhang.github.io/HiMo)
#
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
#
# Description: Self-supervised loss functions.
#
# All losses receive a unified dict from ssl_loss_calculator (trainer.py).
# Every frame is represented only as a List[Tensor] — no flat/offsets/sizes.
#
#   res_dict keys (per frame 'pc0', 'pc1', 'pch1', ...):
#     '{frame}_list'   : List[Tensor (N_i,3)]  one tensor per sample
#     '{frame}_labels' : List[Tensor (N_i,)]   one label vector per sample
#
#   'est_flow_list'    : List[Tensor (N_i,3)]
#   'batch_size'       : int
#   'loss_weights_dict': dict  (teflow* only)
#   'cluster_loss_args': dict  (teflowLoss only)
"""
import torch
from assets.cuda.chamfer3D import nnChamferDis
MyCUDAChamferDis = nnChamferDis()

# NOTE(Qingwen 24/07/06): squared, so it's sqrt(4) = 2m, in 10Hz the vel = 20m/s ~ 72km/h
# If your scenario is different, may need adjust this TRUNCATED to 80-120km/h vel.
TRUNCATED_DIST = 4

# FIXME(Qingwen 25-07-21): hardcoded 10 Hz. Adjust for datasets with different timestamps.
DELTA_T = 0.1  # seconds


# ---- helpers -----------------------------------------------------------------

def get_time_delta(frame_id):
    """Return (time_delta, factor).
    pch1->(-0.1,1), pch2->(-0.2,2), pc1->(+0.1,1), pc2->(+0.2,2)
    """
    if frame_id.startswith('pch'):
        n = int(frame_id[3:]) if len(frame_id) > 3 else 1
        return -DELTA_T * n, n
    elif frame_id.startswith('pc'):
        n = int(frame_id[2:]) if len(frame_id) > 2 else 1
        return DELTA_T * n, n
    raise ValueError(f"Unknown frame ID: {frame_id}")


def _frame_keys(res_dict):
    """Auxiliary frame ids present in res_dict (e.g. ['pc1', 'pch1']), excluding pc0."""
    return [k.replace('_list', '') for k in res_dict
            if k.endswith('_list') \
                and k != 'pc0_list' and k != 'est_flow_list' and not k.endswith('_labels_list')]


# ---- helpers shared by teflow* -----------------------------------------------

def batched_chamfer_related(res_dict, timer=None):
    """Chamfer + dynamic-chamfer over all auxiliary frames via CUDA streams.

    Returns
    -------
    total_chamfer_dis, total_dynamic_chamfer_dis : scalar Tensors
    frame_keys : List[str]
    """
    pc0_list      = res_dict['pc0_list']
    flow_list     = res_dict['est_flow_list']
    pc0_lab_list  = res_dict['pc0_labels_list']
    frame_keys    = _frame_keys(res_dict)
    loss_w        = res_dict['loss_weights_dict']
    chamfer_w     = loss_w.get('chamfer_dis', 0.0)
    dyn_chamfer_w = loss_w.get('dynamic_chamfer_dis', 0.0)

    total_chamfer_dis       = torch.tensor(0.0, device=pc0_list[0].device)
    total_dynamic_chamfer_dis = torch.tensor(0.0, device=pc0_list[0].device)

    for frame_id in frame_keys:
        time_delta, factor = get_time_delta(frame_id)
        weight      = 1.0 if frame_id == 'pc1' else 1.0 / pow(2, factor)
        target_list = res_dict[f'{frame_id}_list']

        # Projected positions: list comprehension keeps everything per-sample
        proj_list = [p0 + (fv / DELTA_T) * time_delta
                     for p0, fv in zip(pc0_list, flow_list)]

        if chamfer_w > 0:
            total_chamfer_dis += MyCUDAChamferDis.batched(
                proj_list, target_list, truncate_dist=TRUNCATED_DIST * factor
            ) * weight

        if dyn_chamfer_w <= 0:
            continue

        tgt_lab_list = res_dict[f'{frame_id}_labels_list']
        proj_dyn, tgt_dyn = [], []
        for proj_i, p0_lab_i, tgt_i, tgt_lab_i in zip(
                proj_list, pc0_lab_list, target_list, tgt_lab_list):
            dp = proj_i[p0_lab_i > 0]
            dt = tgt_i[tgt_lab_i > 0]
            if dp.shape[0] > 256 and dt.shape[0] > 256:
                proj_dyn.append(dp)
                tgt_dyn.append(dt)

        if len(proj_dyn) == 1:
            total_dynamic_chamfer_dis += MyCUDAChamferDis(
                proj_dyn[0], tgt_dyn[0], truncate_dist=TRUNCATED_DIST * factor
            ) * weight
        elif len(proj_dyn) > 1:
            total_dynamic_chamfer_dis += MyCUDAChamferDis.batched(
                proj_dyn, tgt_dyn, truncate_dist=TRUNCATED_DIST * factor
            ) * weight

    n = len(frame_keys)
    if n > 0:
        total_chamfer_dis       /= n
        total_dynamic_chamfer_dis /= n

    return total_chamfer_dis, total_dynamic_chamfer_dis, frame_keys

# ---- multi-frame cluster loss (teflow) -------------------
# Based on TeFlow paper: https://arxiv.org/abs/2602.19053
def multi_frames_clusterLoss(
    pc0_list, pc0_lab_list, flow_list,
    frame_keys, frames_dists, frames_indices, res_dict, args={}
):
    """RANSAC-weighted cluster consistency loss across multiple temporal frames.

    frames_dists[frame_id]   : List[(N_i,)]  per-sample dist from batched_disid_res
    frames_indices[frame_id] : List[(N_i,)]  per-sample LOCAL idx into frame_list[i]
    """
    TOP_K         = int(args.get('top_k_candidates', 5))
    COS_THRESH    = args.get('ransac_cos_threshold', 0.7071)
    TIME_DECAY    = args.get('time_decay_factor', 0.9)
    NET_EST_W     = args.get('network_estimate_weight', 1.0)

    all_cluster_flows, all_target_flows, all_avg_losses = [], [], []

    for i, (p0, lab0, fv) in enumerate(zip(pc0_list, pc0_lab_list, flow_list)):
        for label in torch.unique(lab0):
            if label <= 1:
                continue

            cluster_mask  = (lab0 == label)
            cluster_flows = fv[cluster_mask]

            ext_flows, ext_dists, ext_tw = [], [], []
            for frame_id in frame_keys:
                dist_c = frames_dists[frame_id][i][cluster_mask]
                idx_c  = frames_indices[frame_id][i][cluster_mask]
                if dist_c.shape[0] <= TOP_K:
                    continue
                topk_dists, topk_local = torch.topk(dist_c, k=TOP_K)
                target_pts = res_dict[f'{frame_id}_list'][i][idx_c[topk_local]]
                src_pts    = p0[cluster_mask][topk_local]
                time_delta, factor = get_time_delta(frame_id)
                # Eq. 3 in the TeFlow paper, with time decay and directionality
                flows = (target_pts - src_pts) / factor * (-1 if time_delta < 0 else 1)
                ext_flows.append(flows)
                ext_dists.append(topk_dists)
                ext_tw.append(torch.full((TOP_K,), pow(TIME_DECAY, factor), device=p0.device))

            if not ext_flows:
                continue
            
            # Eq. 2 in the TeFlow paper
            net_avg = cluster_flows.mean(dim=0)
            net_mag = torch.linalg.norm(net_avg)
            # Eq. 4 in the TeFlow paper
            all_cands = torch.cat(ext_flows + [net_avg.unsqueeze(0)], dim=0)
            all_d     = torch.cat(ext_dists + [net_mag.unsqueeze(0)], dim=0)
            all_tw    = torch.cat(ext_tw, dim=0)
            if all_cands.shape[0] < 2:
                continue

            d_norm  = (all_d - all_d.min()) / (all_d.max() - all_d.min() + 1e-6)
            # Eq. 5
            cos_sim = torch.nn.functional.cosine_similarity(
                all_cands[:, None, :], all_cands[None, :, :], dim=-1)
            inlier  = cos_sim > COS_THRESH
            # Eq. 6
            weights = torch.cat([all_tw * (1 + d_norm[:-1]),
                                  (NET_EST_W * (1 + d_norm[-1])).unsqueeze(0)])
            # Eq. 7
            scores  = torch.matmul(inlier.float(), weights.unsqueeze(1)).squeeze()
            best    = torch.argmax(scores)
            
            # Eq. 8
            inlier_flows = all_cands[inlier[best]]
            inlier_w     = weights[inlier[best]]
            denom = inlier_w.sum()
            target_flow = (inlier_w.unsqueeze(1) * inlier_flows).sum(dim=0) / denom \
                          if denom > 1e-6 else all_cands[best]

            all_cluster_flows.append(cluster_flows)
            all_target_flows.append(target_flow.expand_as(cluster_flows))
            all_avg_losses.append(
                torch.linalg.vector_norm(cluster_flows - target_flow, dim=-1).mean()
            )

    # FIXME(Qingwen): maybe afterward we can have weight here to specific different weight on point/cluster etc.
    if not all_cluster_flows:
        return torch.tensor(0.0, device=flow_list[0].device)
    # Eq. 9 with two terms
    # NOTE(Qingwen): Point-level term
    loss  = torch.nn.functional.mse_loss(
        torch.cat(all_cluster_flows), torch.cat(all_target_flows)
    )
    # NOTE(Qingwen): Cluster-level term
    loss += torch.stack(all_avg_losses).mean()
    return loss


# ---- shared cluster loop (seflow / seflowpp) -------------------
# SeFlow Paper: https://arxiv.org/pdf/2407.01702
def _seflow_cluster_loop(pc0_list, pc1_list, pc0_lab_list, pc1_lab_list,
                          flow_list, dist0_list, idx0_list):
    """Per-sample seflow cluster loss (Eq. 6-11).

    dist0_list, idx0_list : output of batched_disid_res(pc0_list, pc1_list)
    idx0_list[i] is LOCAL into pc1_list[i].
    Returns (static_cluster_loss, moved_cluster_loss, have_any_dynamic).
    """
    dev = flow_list[0].device
    static_loss    = torch.tensor(0.0, device=dev)
    cluster_norms  = []
    fallback_dists = []
    have_any_dyn   = False

    for p0, p1, lab0, lab1, fv, dist0, idx0 in zip(
            pc0_list, pc1_list, pc0_lab_list, pc1_lab_list,
            flow_list, dist0_list, idx0_list):
        have_dyn = (lab0 > 0).sum() > 256 and (lab1 > 0).sum() > 256
        if have_dyn:
            have_any_dyn = True
            fallback_dists.append(dist0)

        for label in torch.unique(lab0):
            mask = (lab0 == label)
            if label == 0:
                # Eq. 6 in the paper
                static_loss += torch.linalg.vector_norm(fv[mask], dim=-1).mean()
            elif label > 1 and have_dyn:
                c_flow = fv[mask]
                c_idx0 = idx0[mask]
                # Eq. 8 in the paper
                sorted_local = torch.argsort(dist0[mask], descending=True)
                max_idx = torch.nonzero(lab1[c_idx0[sorted_local]] > 0).squeeze(1)
                if max_idx.shape[0] == 0:
                    continue
                best     = sorted_local[max_idx[0]]
                # Eq. 9 in the paper
                max_flow = p1[c_idx0[best]] - p0[mask][best]
                # Eq. 10 in the paper
                cluster_norms.append(torch.linalg.vector_norm(c_flow - max_flow, dim=-1))

    if cluster_norms:
        # Eq. 11
        moved_loss = torch.cat(cluster_norms).mean()
    elif have_any_dyn:
        all_d = torch.cat(fallback_dists)
        moved_loss = torch.mean(all_d[all_d <= TRUNCATED_DIST])
    else:
        moved_loss = torch.tensor(0.0, device=dev)

    return static_loss, moved_loss

# from paper: https://arxiv.org/abs/2602.19053
def teflowLoss(res_dict, timer=None):
    """Temporal seflow: chamfer over all frames + static + RANSAC cluster loss."""
    pc0_list     = res_dict['pc0_list']
    flow_list    = res_dict['est_flow_list']
    pc0_lab_list = res_dict['pc0_labels_list']

    chamfer_dis, dynamic_chamfer_dis, frame_keys = batched_chamfer_related(res_dict, timer)

    static_loss = torch.tensor(0.0, device=pc0_list[0].device)
    for fv, lab in zip(flow_list, pc0_lab_list):
        if (lab == 0).any():
            static_loss += torch.linalg.vector_norm(fv[lab == 0], dim=-1).mean()
    static_loss /= max(len(pc0_list), 1)

    cluster_weight = res_dict['loss_weights_dict'].get('cluster_based_pc0pc1', 0.0)
    if cluster_weight > 0:
        frames_dists, frames_indices = {}, {}
        for frame_id in frame_keys:
            d_list, i_list = MyCUDAChamferDis.batched_disid_res(
                pc0_list, res_dict[f'{frame_id}_list'],
            )
            frames_dists[frame_id]   = d_list
            frames_indices[frame_id] = i_list

        moved_cluster_loss = multi_frames_clusterLoss(
            pc0_list, pc0_lab_list, flow_list,
            frame_keys, frames_dists, frames_indices, res_dict,
            res_dict.get('cluster_loss_args', {}),
        )
    else:
        moved_cluster_loss = torch.tensor(0.0, device=pc0_list[0].device)

    return {
        'chamfer_dis':          chamfer_dis,
        'dynamic_chamfer_dis':  dynamic_chamfer_dis,
        'static_flow_loss':     static_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
    }

# from paper: https://arxiv.org/abs/2503.00803
def seflowppLoss(res_dict, timer=None):
    """seflow++ loss: bidirectional (pc1 + pch1) chamfer + cluster, B samples."""
    pc0_list      = res_dict['pc0_list']
    pc1_list      = res_dict['pc1_list']
    pch1_list     = res_dict['pch1_list']
    flow_list     = res_dict['est_flow_list']
    pc0_lab_list  = res_dict['pc0_labels_list']
    pc1_lab_list  = res_dict['pc1_labels_list']
    pch1_lab_list = res_dict['pch1_labels_list']
    dev           = pc0_list[0].device

    fwd_list  = [p0 + fv for p0, fv in zip(pc0_list, flow_list)]
    bwd_list  = [p0 - fv for p0, fv in zip(pc0_list, flow_list)]

    # Chamfer: both temporal directions concurrently
    chamfer_dis  = MyCUDAChamferDis.batched(fwd_list, pc1_list,  truncate_dist=TRUNCATED_DIST)
    chamfer_dis += MyCUDAChamferDis.batched(bwd_list, pch1_list, truncate_dist=TRUNCATED_DIST)

    # Dynamic chamfer
    dyn_fwd, dyn_pc1   = [], []
    dyn_bwd, dyn_pch1  = [], []
    for fwd_i, bwd_i, p1_i, ph1_i, lab0_i, lab1_i, labh1_i in zip(
            fwd_list, bwd_list, pc1_list, pch1_list,
            pc0_lab_list, pc1_lab_list, pch1_lab_list):
        dyn_mask = lab0_i > 0
        if dyn_mask.sum() > 256:
            dp1 = p1_i[lab1_i > 0]
            dph = ph1_i[labh1_i > 0]
            if dp1.shape[0]  > 256: dyn_fwd.append(fwd_i[dyn_mask]); dyn_pc1.append(dp1)
            if dph.shape[0]  > 256: dyn_bwd.append(bwd_i[dyn_mask]); dyn_pch1.append(dph)

    dynamic_chamfer_dis = torch.tensor(0.0, device=dev)
    if len(dyn_fwd) == 1:
        dynamic_chamfer_dis += MyCUDAChamferDis(dyn_fwd[0], dyn_pc1[0], truncate_dist=TRUNCATED_DIST)
    elif len(dyn_fwd) > 1:
        dynamic_chamfer_dis += MyCUDAChamferDis.batched(dyn_fwd, dyn_pc1, truncate_dist=TRUNCATED_DIST)
    if len(dyn_bwd) == 1:
        dynamic_chamfer_dis += MyCUDAChamferDis(dyn_bwd[0], dyn_pch1[0], truncate_dist=TRUNCATED_DIST)
    elif len(dyn_bwd) > 1:
        dynamic_chamfer_dis += MyCUDAChamferDis.batched(dyn_bwd, dyn_pch1, truncate_dist=TRUNCATED_DIST)

    dist0_list, idx0_list = MyCUDAChamferDis.batched_disid_res(pc0_list, pc1_list)
    static_loss, moved_cluster_loss = _seflow_cluster_loop(
        pc0_list, pc1_list, pc0_lab_list, pc1_lab_list,
        flow_list, dist0_list, idx0_list,
    )

    return {
        'chamfer_dis':          chamfer_dis / 2.0,
        'dynamic_chamfer_dis':  dynamic_chamfer_dis / 2.0,
        'static_flow_loss':     static_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
    }

# from paper: https://arxiv.org/abs/2407.01702
def seflowLoss(res_dict, timer=None):
    """seflow loss: single future frame (pc1), batched over B samples."""
    pc0_list     = res_dict['pc0_list']
    pc1_list     = res_dict['pc1_list']
    flow_list    = res_dict['est_flow_list']
    pc0_lab_list = res_dict['pc0_labels_list']
    pc1_lab_list = res_dict['pc1_labels_list']
    dev          = pc0_list[0].device

    fwd_list = [p0 + fv for p0, fv in zip(pc0_list, flow_list)]

    chamfer_dis = MyCUDAChamferDis.batched(fwd_list, pc1_list, truncate_dist=TRUNCATED_DIST)

    # Dynamic chamfer
    dyn_fwd, dyn_pc1 = [], []
    for fwd_i, p1_i, lab0_i, lab1_i in zip(fwd_list, pc1_list, pc0_lab_list, pc1_lab_list):
        dp1 = p1_i[lab1_i > 0]
        if (lab0_i > 0).sum() > 256 and dp1.shape[0] > 256:
            dyn_fwd.append(fwd_i[lab0_i > 0])
            dyn_pc1.append(dp1)

    dynamic_chamfer_dis = torch.tensor(0.0, device=dev)
    if len(dyn_fwd) == 1:
        dynamic_chamfer_dis = MyCUDAChamferDis(dyn_fwd[0], dyn_pc1[0], truncate_dist=TRUNCATED_DIST)
    elif len(dyn_fwd) > 1:
        dynamic_chamfer_dis = MyCUDAChamferDis.batched(dyn_fwd, dyn_pc1, truncate_dist=TRUNCATED_DIST)

    dist0_list, idx0_list = MyCUDAChamferDis.batched_disid_res(pc0_list, pc1_list)
    static_loss, moved_cluster_loss = _seflow_cluster_loop(
        pc0_list, pc1_list, pc0_lab_list, pc1_lab_list,
        flow_list, dist0_list, idx0_list,
    )

    return {
        'chamfer_dis':          chamfer_dis,
        'dynamic_chamfer_dis':  dynamic_chamfer_dis,
        'static_flow_loss':     static_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
    }