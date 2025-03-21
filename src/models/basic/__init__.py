import torch
import torch.nn as nn

@torch.no_grad()
def cal_pose0to1(pose0: torch.Tensor, pose1: torch.Tensor):
    """
    Note(Qingwen 2023-12-05 11:09):
    Don't know why but it needed set the pose to float64 to calculate the inverse 
    otherwise it will be not expected result....
    """
    pose1_inv = torch.eye(4, dtype=torch.float64, device=pose1.device)
    pose1_inv[:3,:3] = pose1[:3,:3].T
    pose1_inv[:3,3] = (pose1[:3,:3].T * -pose1[:3,3]).sum(axis=1)
    pose_0to1 = pose1_inv @ pose0.type(torch.float64)
    return pose_0to1.type(torch.float32)

def wrap_batch_pcs(batch, num_frames=2):
    batch_sizes = len(batch["pose0"])

    pose_flows = []
    transform_pc0s = []
    transform_pc_m_frames = [[] for _ in range(num_frames - 2)]
    # print(batch)
    for batch_id in range(batch_sizes):
        selected_pc0 = batch["pc0"][batch_id] 
        with torch.no_grad():
            if 'ego_motion' in batch:
                pose_0to1 = batch['ego_motion'][batch_id].type(torch.float32)
            else:
                pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id]) 
            if num_frames > 2: 
                past_poses = []
                for i in range(1, num_frames - 1):
                    past_pose = cal_pose0to1(batch[f"poseh{i}"][batch_id], batch["pose1"][batch_id])
                    past_poses.append(past_pose)

        transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3] #t -> t+1 warping

        pose_flows.append(transform_pc0 - selected_pc0)
        transform_pc0s.append(transform_pc0)

        for i in range(1, num_frames - 1):
            selected_pc_m = batch[f"pch{i}"][batch_id]
            transform_pc_m = selected_pc_m @ past_poses[i-1][:3, :3].T + past_poses[i-1][:3, 3]
            transform_pc_m_frames[i-1].append(transform_pc_m)

    pc_m_frames = [torch.stack(transform_pc_m_frames[i], dim=0) for i in range(num_frames - 2)]

    pc0s = torch.stack(transform_pc0s, dim=0) 
    pc1s = batch["pc1"]
    pcs_dict = {
        'pc0s': pc0s,
        'pc1s': pc1s,
        'pose_flows': pose_flows
    }
    for i in range(1, num_frames - 1):
        pcs_dict[f'pch{i}s'] = pc_m_frames[i-1]
    
    return pcs_dict

class ConvWithNorms(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size,
                              stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_num_channels)
        self.nonlinearity = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.nonlinearity(batchnorm_res)