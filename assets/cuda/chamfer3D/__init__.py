"""
# Created: 2023-08-04 11:20
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of SeFlow (https://github.com/KTH-RPL/SeFlow).
# If you find this repo helpful, please cite the respective publication as
# listed on the above website.
#
# Description: ChamferDis speedup using CUDA.
#
# NOTE(2026-03-11, Qingwen) Why CUDA streams (not batched kernel):
#   At N=88K pts/sample on RTX 3090, one sample already uses 4.2 SM waves,
#   so any kernel-level batching hits the same hardware ceiling.
#   Streams give ~1.14× speedup by overlapping B independent kernel launches.
#   More importantly, they keep the GPU busy with fewer CPU-GPU sync gaps,
#   preventing GPU utilization from spiking which triggers cluster job kills.
#
"""
from __future__ import annotations

from torch import nn
from torch.autograd import Function
import torch, os, time
from typing import List
import chamfer3D

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


class ChamferDis(Function):
    """Single-sample Chamfer distance: pc0 (N,3) × pc1 (M,3) on GPU."""

    @staticmethod
    def forward(ctx, pc0, pc1):
        dis0 = torch.zeros(pc0.shape[0], device=pc0.device).contiguous()
        dis1 = torch.zeros(pc1.shape[0], device=pc1.device).contiguous()
        idx0 = torch.zeros(pc0.shape[0], dtype=torch.int32, device=pc0.device).contiguous()
        idx1 = torch.zeros(pc1.shape[0], dtype=torch.int32, device=pc1.device).contiguous()
        chamfer3D.forward(pc0, pc1, dis0, dis1, idx0, idx1)
        ctx.save_for_backward(pc0, pc1, idx0, idx1)
        return dis0, dis1, idx0, idx1

    @staticmethod
    def backward(ctx, gd0, gd1, _gi0, _gi1):
        pc0, pc1, idx0, idx1 = ctx.saved_tensors
        gpc0 = torch.zeros_like(pc0)
        gpc1 = torch.zeros_like(pc1)
        chamfer3D.backward(pc0, pc1, idx0, idx1,
                           gd0.contiguous(), gd1.contiguous(), gpc0, gpc1)
        return gpc0, gpc1

# ─── nn.Module ────────────────────────────────────────────────────────────────
class nnChamferDis(nn.Module):
    """Chamfer distance loss — single and batched-via-streams modes.

    Methods
    -------
    forward(pc0, pc1)
        Single-sample or list-of-samples loss. Used by seflowLoss / seflowppLoss.
        If a list is provided, it processes it in parallel via CUDA streams.

    dis_res(pc0, pc1)        → (dist0, dist1), no reduction
    disid_res(pc0, pc1)      → (dist0, dist1, idx0, idx1), no reduction
    truncated_dis(pc0, pc1)  → NSFP-style truncated loss
    """

    def __init__(self, truncate_dist: bool = True):
        super().__init__()
        self.truncate_dist = truncate_dist
        # Pre-allocate streams once to avoid per-call creation overhead (~50 µs each)
        self._streams: List[torch.cuda.Stream] = []

    def _ensure_streams(self, n: int) -> List[torch.cuda.Stream]:
        while len(self._streams) < n:
            self._streams.append(torch.cuda.Stream())
        return self._streams[:n]

    # ── forward ─────────────────────────────────────────────────

    def forward(self, input0, input1, truncate_dist: float = -1, **_ignored) -> torch.Tensor:
        """Chamfer loss. Supports single tensor or list of tensors."""
        if not isinstance(input0, list):
            dist0, dist1, _, _ = ChamferDis.apply(input0.contiguous(), input1.contiguous())
            if truncate_dist <= 0:
                return dist0.mean() + dist1.mean()
            v0, v1 = dist0 <= truncate_dist, dist1 <= truncate_dist
            return torch.nanmean(dist0[v0]) + torch.nanmean(dist1[v1])

        # Batched processing via CUDA streams
        B = len(input0)
        if B == 1:
            return self.forward(input0[0], input1[0], truncate_dist)

        streams  = self._ensure_streams(B)
        main     = torch.cuda.current_stream()
        per_loss: List[torch.Tensor] = [None] * B  # type: ignore[list-item]

        for i in range(B):
            streams[i].wait_stream(main)
            with torch.cuda.stream(streams[i]):
                d0, d1, _, _ = ChamferDis.apply(input0[i].contiguous(),
                                                 input1[i].contiguous())
                if truncate_dist <= 0:
                    per_loss[i] = d0.mean() + d1.mean()
                else:
                    v0, v1 = d0 <= truncate_dist, d1 <= truncate_dist
                    per_loss[i] = torch.nanmean(d0[v0]) + torch.nanmean(d1[v1])

        for i in range(B):
            main.wait_stream(streams[i])

        return torch.stack(per_loss).mean()

    # ── batched disid_res via CUDA streams (for cluster precomputation) ───────

    def batched_disid_res(self,
                          pc0_list: List[torch.Tensor],
                          pc1_list: List[torch.Tensor],
                          ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Parallel disid_res across B samples via CUDA streams.

        Same list-in / list-out convention as batched().

        Returns
        -------
        dist0_list : List[(N_i,)]  per-point nearest distance in pc1_i
        idx0_list  : List[(N_i,)]  LOCAL index into pc1_list[i]  (0 .. M_i-1)

        Usage:
            dist0_list, idx0_list = fn.batched_disid_res(pc0_list, pc1_list)
            neighbour = pc1_list[i][idx0_list[i][mask]]   # no global arithmetic
        """
        B = len(pc0_list)
        if B == 1:
            d0, _, i0, _ = ChamferDis.apply(pc0_list[0].contiguous(), pc1_list[0].contiguous())
            return [d0], [i0]

        streams = self._ensure_streams(B)
        main    = torch.cuda.current_stream()
        d0_list: List[torch.Tensor] = [None] * B  # type: ignore[list-item]
        i0_list: List[torch.Tensor] = [None] * B  # type: ignore[list-item]

        for i in range(B):
            streams[i].wait_stream(main)
            with torch.cuda.stream(streams[i]):
                d0, _, idx0, _ = ChamferDis.apply(pc0_list[i].contiguous(),
                                                   pc1_list[i].contiguous())
                d0_list[i] = d0
                i0_list[i] = idx0   # local index — no offset arithmetic needed

        for i in range(B):
            main.wait_stream(streams[i])

        return d0_list, i0_list

    # ── utilities ─────────────────────────────────────────────────────────────

    def dis_res(self, input0: torch.Tensor, input1: torch.Tensor):
        """Return raw (dist0, dist1) without reduction."""
        d0, d1, _, _ = ChamferDis.apply(input0.contiguous(), input1.contiguous())
        return d0, d1

    def disid_res(self, input0: torch.Tensor, input1: torch.Tensor):
        """Return raw (dist0, dist1, idx0, idx1) without reduction."""
        return ChamferDis.apply(input0.contiguous(), input1.contiguous())

    def truncated_dis(self, input0: torch.Tensor, input1: torch.Tensor,
                      truncate_dist: float = 2.0) -> torch.Tensor:
        """NSFP-style: distances >= threshold clamped to 0, then mean."""
        cx, cy = self.dis_res(input0, input1)
        cx[cx >= truncate_dist] = 0.0
        cy[cy >= truncate_dist] = 0.0
        return cx.mean() + cy.mean()


if __name__ == "__main__":
    import numpy as np
    pc0_np = np.load(f'{BASE_DIR}/tests/test_pc0.npy')[..., :3]
    pc1_np = np.load(f'{BASE_DIR}/tests/test_pc1.npy')[..., :3]
    pc0 = torch.from_numpy(pc0_np).float().cuda()
    pc1 = torch.from_numpy(pc1_np).float().cuda()
    fn  = nnChamferDis(truncate_dist=False)

    loss_s = fn(pc0, pc1)
    print(f"Single:          {loss_s.item():.6f}")

    for B in [2, 4, 8]:
        lb = fn([pc0.clone()]*B, [pc1.clone()]*B)
        print(f"Batched   B={B}: {lb.item():.6f}  {'✓' if torch.allclose(loss_s, lb, atol=1e-5) else '✗'}")

    # Test batched_disid_res global indexing
    print("\n--- batched_disid_res global index test ---")
    B = 2
    pc0_b = torch.cat([pc0]*B)
    pc1_b = torch.cat([pc1]*B)
    N0, N1 = pc0.shape[0], pc1.shape[0]
    offs0 = torch.tensor([0, N0], dtype=torch.int32, device='cuda')
    szs0  = torch.tensor([N0, N0], dtype=torch.int32, device='cuda')
    offs1 = torch.tensor([0, N1], dtype=torch.int32, device='cuda')
    szs1  = torch.tensor([N1, N1], dtype=torch.int32, device='cuda')
    pc0_lst = [pc0]*B
    pc1_lst = [pc1]*B
    d0_lst_out, i0_lst_out = fn.batched_disid_res(pc0_lst, pc1_lst)
    assert len(d0_lst_out) == B and len(i0_lst_out) == B, "wrong list length"
    for j in range(B):
        assert (i0_lst_out[j] < N1).all(), f"sample-{j} idx out of range"
    print("Local index check: ✓")