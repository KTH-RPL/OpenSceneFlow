import torch
import pytorch3d.ops as pytorch3d_ops

def calculate_unq_voxels(coords, image_dims):
    unqs, idxs = torch.unique(coords[:, 1]*image_dims[0]+coords[:, 2], return_inverse=True, sorted=True)
    unqs_voxel = torch.stack([unqs//image_dims[0], unqs%image_dims[0]], dim=1)
    return unqs_voxel, idxs 

# https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/utils.html
def batched_masked_gather(x: torch.Tensor, idxs: torch.Tensor, mask: torch.Tensor,fill_value=-1.0) -> torch.Tensor:
    assert x.dim() == 3 # [b, m, c]
    assert idxs.dim() == 3 # [b, n, k]
    assert idxs.shape == mask.shape
    b, m, c = x.shape

    idxs_masked = idxs.clone()
    idxs_masked[~mask] = 0
    l, n, k = idxs.shape
    y = pytorch3d_ops.knn_gather(x, idxs_masked) # [b, n, k, c]
    y[~mask, :] = fill_value

    return y

def pad_to_batch(idxs, l):
    if idxs is None:
        return idxs
    if idxs.dim()==2:
        assert idxs.shape[0]<=l
        if idxs.shape[0]<l:
            padding = torch.zeros((l-idxs.shape[0], idxs.shape[1]), device=idxs.device)-1
            idxs = torch.cat([idxs, padding], dim=0)
        else: 
            pass
    elif idxs.dim()==1:
        assert idxs.shape[0]<=l
        if idxs.shape[0]<l:
            padding = torch.zeros((l-idxs.shape[0]), device=idxs.device)-1
            idxs = torch.cat([idxs, padding], dim=0)
        else: 
            pass
    else: 
        NotImplementedError

    return idxs

