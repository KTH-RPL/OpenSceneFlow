
"""
This file is from: https://github.com/yanconglin/ICP-Flow
with slightly modification to have unified format with all benchmark.
"""

import torch
import os, sys, time, fire
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..' ))
sys.path.append(BASE_DIR)

import pytorch3d.transforms as pytorch3d_t
import numpy as np
import pytorch3d.ops as p3d

def nearest_neighbor_batch(src, dst):
    assert src.dim()==3
    assert dst.dim()==3
    assert len(src)==len(dst)
    b, num, dim = src.shape
    assert src.shape[2]>=3
    assert dst.shape[2]>=3
    result = p3d.knn_points(src[:, :, 0:3], dst[:, :, 0:3], K=1)
    dst_idxs = result.idx
    distances = result.dists
    return dst_idxs.view(b, num), distances.view(b, num).sqrt()

# modified the original pytorch3d icp
# check the dif between open3d icp and pytorch3d icp
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple, TYPE_CHECKING, Union
import warnings
from typing import List, NamedTuple, Optional, TYPE_CHECKING, Union

import torch
from pytorch3d.ops import knn_points
from pytorch3d.ops import utils as oputil
from pytorch3d.structures import utils as strutil
if TYPE_CHECKING:
    from pytorch3d.structures.pointclouds import Pointclouds

# named tuples for inputs/outputs
class SimilarityTransform(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    s: torch.Tensor


class ICPSolution(NamedTuple):
    converged: bool
    rmse: Union[torch.Tensor, None]
    Xt: torch.Tensor
    RTs: SimilarityTransform
    t_history: List[SimilarityTransform]


def iterative_closest_point(
    X: Union[torch.Tensor, "Pointclouds"],
    Y: Union[torch.Tensor, "Pointclouds"],
    init_transform: Optional[SimilarityTransform] = None,
    thres: float = 0.1, 
    max_iterations: int = 100,
    relative_rmse_thr: float = 1e-6,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    verbose: bool = False,
) -> ICPSolution:
    """
    Executes the iterative closest point (ICP) algorithm [1, 2] in order to find
    a similarity transformation (rotation `R`, translation `T`, and
    optionally scale `s`) between two given differently-sized sets of
    `d`-dimensional points `X` and `Y`, such that:

    `s[i] X[i] R[i] + T[i] = Y[NN[i]]`,

    for all batch indices `i` in the least squares sense. Here, Y[NN[i]] stands
    for the indices of nearest neighbors from `Y` to each point in `X`.
    Note, however, that the solution is only a local optimum.

    Args:
        **X**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_X, d)` or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points
            of shape `(minibatch, num_points_Y, d)` or a `Pointclouds` object.
        **init_transform**: A named-tuple `SimilarityTransform` of tensors
            `R`, `T, `s`, where `R` is a batch of orthonormal matrices of
            shape `(minibatch, d, d)`, `T` is a batch of translations
            of shape `(minibatch, d)` and `s` is a batch of scaling factors
            of shape `(minibatch,)`.
        **max_iterations**: The maximum number of ICP iterations.
        **relative_rmse_thr**: A threshold on the relative root mean squared error
            used to terminate the algorithm.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes the identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **verbose**: If `True`, prints status messages during each ICP iteration.

    Returns:
        A named tuple `ICPSolution` with the following fields:
        **converged**: A boolean flag denoting whether the algorithm converged
            successfully (=`True`) or not (=`False`).
        **rmse**: Attained root mean squared error after termination of ICP.
        **Xt**: The point cloud `X` transformed with the final transformation
            (`R`, `T`, `s`). If `X` is a `Pointclouds` object, returns an
            instance of `Pointclouds`, otherwise returns `torch.Tensor`.
        **RTs**: A named tuple `SimilarityTransform` containing
        a batch of similarity transforms with fields:
            **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
            **T**: Batch of translations of shape `(minibatch, d)`.
            **s**: batch of scaling factors of shape `(minibatch, )`.
        **t_history**: A list of named tuples `SimilarityTransform`
            the transformation parameters after each ICP iteration.

    References:
        [1] Besl & McKay: A Method for Registration of 3-D Shapes. TPAMI, 1992.
        [2] https://en.wikipedia.org/wiki/Iterative_closest_point
    """
    Xt = X[:, :, 0:3]
    Yt = Y[:, :, 0:3]
    b, size_X, dim = Xt.shape
    if (Xt.shape[2] != Yt.shape[2]) or (Xt.shape[0] != Yt.shape[0]):
        raise ValueError(
            "Point sets X and Y have to have the same "
            + "number of batches and data dimensions."
        )

    mask_X = X[:, :, -1]>0.0
    mask_Y = Y[:, :, -1]>0.0
    num_points_X = mask_X.sum(dim=-1)
    num_points_Y = mask_Y.sum(dim=-1)

    # clone the initial point cloud
    Xt_init = Xt.clone()
    mask_X_init = mask_X.clone()

    if init_transform is not None:
        # parse the initial transform from the input and apply to Xt
        try:
            R, T, s = init_transform
            assert (
                R.shape == torch.Size((b, dim, dim))
                and T.shape == torch.Size((b, dim))
                and s.shape == torch.Size((b,))
            )
        except Exception:
            raise ValueError(
                "The initial transformation init_transform has to be "
                "a named tuple SimilarityTransform with elements (R, T, s). "
                "R are dim x dim orthonormal matrices of shape "
                "(minibatch, dim, dim), T is a batch of dim-dimensional "
                "translations of shape (minibatch, dim) and s is a batch "
                "of scalars of shape (minibatch,)."
            ) from None
        # apply the init transform to the input point cloud
        Xt = _apply_similarity_transform(Xt, R, T, s)
    else:
        # initialize the transformation with identity
        R = oputil.eyes(dim, b, device=Xt.device, dtype=Xt.dtype)
        T = Xt.new_zeros((b, dim))
        s = Xt.new_ones(b)

    prev_rmse = None
    rmse = None
    iteration = -1
    converged = False

    # initialize the transformation history
    t_history = []

    # the main loop over ICP iterations
    for iteration in range(max_iterations):
        knn_result = knn_points(
            Xt, Yt, lengths1=num_points_X, lengths2=num_points_Y, K=1, return_nn=True, norm=2
        )
        Xt_nn_points = knn_result.knn[:, :, 0, :]
        # print('knn points: ', valid.shape, Xt_init.shape, Xt_nn_points.shape)

        valid = knn_result.dists[:, :, 0]<=thres**2
        mask_X = torch.logical_and(mask_X_init, valid)

        X1 = Xt_init * mask_X[:, :, None]
        X2 = Xt_nn_points * mask_X[:, :, None]
        # print('knn points: ', valid.shape, mask_X.shape, Xt_nn_points.shape)

        # get the alignment of the nearest neighbors from Yt with Xt
        R, T, s = corresponding_points_alignment(
            X1,
            X2,
            weights=mask_X,
            estimate_scale=estimate_scale,
            allow_reflection=allow_reflection,
        )

        # apply the estimated similarity transform to Xt_init
        Xt = _apply_similarity_transform(Xt_init, R, T, s)
        # if iteration%2==0: 
        #     src = Xt[0]
        #     dst = Yt[0]
        #     visualize_pcd(np.concatenate([src.cpu().numpy(), dst.cpu().numpy()], axis=0), 
        #                   np.concatenate([np.zeros((len(src)))+1, np.zeros((len(dst)))+2], axis=0), 
        #                   num_colors=3,
        #                   title=f'registration, iter: {iteration} {len(src)} vs {len(dst)}')

        # add the current transformation to the history
        t_history.append(SimilarityTransform(R, T, s))

        # compute the root mean squared error
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        Xt_sq_diff = ((Xt - Xt_nn_points) ** 2).sum(2)
        rmse = oputil.wmean(Xt_sq_diff[:, :, None], mask_X).sqrt()[:, 0, 0]

        # compute the relative rmse
        if prev_rmse is None:
            relative_rmse = rmse.new_ones(b)
        else:
            relative_rmse = (prev_rmse - rmse) / prev_rmse

        if verbose:
            rmse_msg = (
                f"ICP iteration {iteration}: mean/max rmse = "
                + f"{rmse.mean():1.2e}/{rmse.max():1.2e} "
                + f"; mean relative rmse = {relative_rmse.mean():1.2e}"
            )
            print(rmse_msg)

        # check for convergence
        if (relative_rmse <= relative_rmse_thr).all():
            converged = True
            break

        # update the previous rmse
        prev_rmse = rmse

    if verbose:
        if converged:
            print(f"ICP has converged in {iteration + 1} iterations.")
        else:
            print(f"ICP has not converged in {max_iterations} iterations.")

    if oputil.is_pointclouds(X):
        Xt = X.update_padded(Xt)  # type: ignore

    return ICPSolution(converged, rmse, Xt, SimilarityTransform(R, T, s), t_history)


# threshold for checking that point crosscorelation
# is full rank in corresponding_points_alignment
AMBIGUOUS_ROT_SINGULAR_THR = 1e-15


def corresponding_points_alignment(
    X: Union[torch.Tensor, "Pointclouds"],
    Y: Union[torch.Tensor, "Pointclouds"],
    weights: Union[torch.Tensor, List[torch.Tensor], None] = None,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
    eps: float = 1e-9,
) -> SimilarityTransform:
    """
    Finds a similarity transformation (rotation `R`, translation `T`
    and optionally scale `s`)  between two given sets of corresponding
    `d`-dimensional points `X` and `Y` such that:

    `s[i] X[i] R[i] + T[i] = Y[i]`,

    for all batch indexes `i` in the least squares sense.

    The algorithm is also known as Umeyama [1].

    Args:
        **X**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **Y**: Batch of `d`-dimensional points of shape `(minibatch, num_point, d)`
            or a `Pointclouds` object.
        **weights**: Batch of non-negative weights of
            shape `(minibatch, num_point)` or list of `minibatch` 1-dimensional
            tensors that may have different shapes; in that case, the length of
            i-th tensor should be equal to the number of points in X_i and Y_i.
            Passing `None` means uniform weights.
        **estimate_scale**: If `True`, also estimates a scaling component `s`
            of the transformation. Otherwise assumes an identity
            scale and returns a tensor of ones.
        **allow_reflection**: If `True`, allows the algorithm to return `R`
            which is orthonormal but has determinant==-1.
        **eps**: A scalar for clamping to avoid dividing by zero. Active for the
            code that estimates the output scale `s`.

    Returns:
        3-element named tuple `SimilarityTransform` containing
        - **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
        - **T**: Batch of translations of shape `(minibatch, d)`.
        - **s**: batch of scaling factors of shape `(minibatch, )`.

    References:
        [1] Shinji Umeyama: Least-Suqares Estimation of
        Transformation Parameters Between Two Point Patterns
    """

    # make sure we convert input Pointclouds structures to tensors
    Xt, num_points = oputil.convert_pointclouds_to_tensor(X)
    Yt, num_points_Y = oputil.convert_pointclouds_to_tensor(Y)

    if (Xt.shape != Yt.shape) or (num_points != num_points_Y).any():
        raise ValueError(
            "Point sets X and Y have to have the same \
            number of batches, points and dimensions."
        )
    if weights is not None:
        if isinstance(weights, list):
            if any(np != w.shape[0] for np, w in zip(num_points, weights)):
                raise ValueError(
                    "number of weights should equal to the "
                    + "number of points in the point cloud."
                )
            weights = [w[..., None] for w in weights]
            weights = strutil.list_to_padded(weights)[..., 0]

        if Xt.shape[:2] != weights.shape:
            raise ValueError("weights should have the same first two dimensions as X.")

    b, n, dim = Xt.shape

    if (num_points < Xt.shape[1]).any() or (num_points < Yt.shape[1]).any():
        # in case we got Pointclouds as input, mask the unused entries in Xc, Yc
        mask = (
            torch.arange(n, dtype=torch.int64, device=Xt.device)[None]
            < num_points[:, None]
        ).type_as(Xt)
        weights = mask if weights is None else mask * weights.type_as(Xt)

    # compute the centroids of the point sets
    Xmu = oputil.wmean(Xt, weight=weights, eps=eps)
    Ymu = oputil.wmean(Yt, weight=weights, eps=eps)

    # mean-center the point sets
    Xc = Xt - Xmu
    Yc = Yt - Ymu

    total_weight = torch.clamp(num_points, 1)
    # special handling for heterogeneous point clouds and/or input weights
    if weights is not None:
        Xc *= weights[:, :, None]
        Yc *= weights[:, :, None]
        total_weight = torch.clamp(weights.sum(1), eps)

    if (num_points < (dim + 1)).any():
        warnings.warn(
            "The size of one of the point clouds is <= dim+1. "
            + "corresponding_points_alignment cannot return a unique rotation."
        )

    # compute the covariance XYcov between the point sets Xc, Yc
    XYcov = torch.bmm(Xc.transpose(2, 1), Yc)
    XYcov = XYcov / total_weight[:, None, None]

    # decompose the covariance matrix XYcov
    U, S, V = torch.svd(XYcov)

    # catch ambiguous rotation by checking the magnitude of singular values
    # if (S.abs() <= AMBIGUOUS_ROT_SINGULAR_THR).any() and not (
    #     num_points < (dim + 1)
    # ).any():
    #     warnings.warn(
    #         "Excessively low rank of "
    #         + "cross-correlation between aligned point clouds. "
    #         + "corresponding_points_alignment cannot return a unique rotation."
    #     )

    # identity matrix used for fixing reflections
    E = torch.eye(dim, dtype=XYcov.dtype, device=XYcov.device)[None].repeat(b, 1, 1)

    if not allow_reflection:
        # reflection test:
        #   checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # find the rotation matrix by composing U and V again
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # estimate the scaling component of the transformation
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        Xcov = (Xc * Xc).sum((1, 2)) / total_weight

        # the scaling component
        s = trace_ES / torch.clamp(Xcov, eps)

        # translation component
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        # translation component
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]

        # unit scaling since we do not estimate scale
        s = T.new_ones(b)

    return SimilarityTransform(R, T, s)


def _apply_similarity_transform(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X

# -----------------------------------------------------------------------------------------------------------
def setdiff1d(t1, t2):
    # indices = torch.new_ones(t1.shape, dtype = torch.uint8)
    # for elem in t2:
    #     indices = indices & (t1 != elem)  
    # intersection = t1[indices]  
    # assuming t2 is a subset of t1
    t1_unique = torch.unique(t1)
    t2_unique = torch.unique(t2)
    assert len(t1_unique)>=len(t2_unique)
    t12, counts = torch.cat([t1_unique, t2_unique]).unique(return_counts=True)
    diff = t12[torch.where(counts.eq(1))]
    return diff

def get_bbox_tensor(points):
    x = torch.abs(points[:, 0].max() - points[:, 0].min())
    y = torch.abs(points[:, 1].max() - points[:, 1].min())
    z = torch.abs(points[:, 2].max() - points[:, 2].min()) 
    return sorted([x, y, z])

def sanity_check(args, src_points, dst_points, src_labels, dst_labels, pairs):
    pairs_true = []
    for pair in pairs:
        src = src_points[src_labels==pair[0]]
        dst = dst_points[dst_labels==pair[1]]
        # print('sanity check :', pair, len(src), len(dst), src.mean(0), dst.mean(0), torch.linalg.norm(dst.mean(0) - src.mean(0)), args.translation_frame)

        # scenario 1: either src or dst does not exist, return None
        # scenario 2: both src or dst exist, but they are not matchable because of ground points/too few points/size mismatch, return False
        # scenario 3: both src or dst exist, and they are are matchable, return True
        if min(len(src), len(dst))<args.min_cluster_size: continue
        if min(pair[0], pair[1])<0: continue  # ground or non-clustered points

        mean_src = src.mean(0)
        mean_dst = dst.mean(0)
        if torch.linalg.norm((mean_dst - mean_src)[0:2])>args.translation_frame: continue # x/y translation

        src_bbox = get_bbox_tensor(src)
        dst_bbox = get_bbox_tensor(dst)
        # print('sanity check bbox:', src_bbox, dst_bbox)
        if min(src_bbox[0], dst_bbox[0]) < args.thres_box * max(src_bbox[0], dst_bbox[0]): continue 
        if min(src_bbox[1], dst_bbox[1]) < args.thres_box * max(src_bbox[1], dst_bbox[1]): continue 
        if min(src_bbox[2], dst_bbox[2]) < args.thres_box * max(src_bbox[2], dst_bbox[2]): continue 

        pairs_true.append(pair)
    if len(pairs_true)>0:
        return torch.vstack(pairs_true)
    else:
        return torch.zeros((0,2))
    
def match_pcds(args, src_points, dst_points, src_labels, dst_labels):

    src_labels_unq = torch.unique(src_labels, return_counts=False).long()
    dst_labels_unq = torch.unique(dst_labels, return_counts=False).long()
    labels_unq = torch.unique(torch.cat([src_labels_unq, dst_labels_unq], axis=0), return_counts=False)

    # # # stage 1: match static: overlapped clusters
    pairs = torch.stack([labels_unq, labels_unq], dim=1)
    mask = pairs.min(dim=1)[0]>=0 # remove ground
    pairs = pairs[mask]
    pairs_true = sanity_check(args, src_points, dst_points, src_labels, dst_labels, pairs)
    # print('sanity check: sta: ', len(pairs), len(pairs_true), pairs_true)

    if len(pairs_true)>0: 
        pairs_sta, transformations_sta = match_pairs(args, src_points, dst_points, src_labels, dst_labels, pairs_true) 
    else:
        pairs_sta, transformations_sta = torch.tensor([]).reshape(0, 10), torch.tensor([]).reshape(0, 4, 4)
    # print('pairs_sta: ', len(pairs), len(pairs_sta), pairs_sta[:, 0:2])

    # stage 2: match dynamic
    # # # remove matched near-static pairs:
    if len(pairs_sta)<len(labels_unq):
        if len(pairs_sta)>0:
            src_labels_unq = setdiff1d(src_labels_unq, pairs_sta[:, 0])
            dst_labels_unq = setdiff1d(dst_labels_unq, pairs_sta[:, 1])

        pairs = torch.stack([src_labels_unq.repeat_interleave(len(dst_labels_unq)), dst_labels_unq.repeat(len(src_labels_unq))], dim=1)
        pairs_true = sanity_check(args, src_points, dst_points, src_labels, dst_labels, pairs)
    else:
        pairs_true = torch.zeros(0, 2)
    # print('dynamic src_labels, dst_labels: ', src_labels_unq.long(), dst_labels_unq.long(), pairs_true)

    if len(pairs_true)>0: 
        pairs_dyn, transformations_dyn = match_pairs(args, src_points, dst_points, src_labels, dst_labels, pairs_true) 
        # print('dynamic paired_idxs: ', len(pairs), len(pairs_true), len(pairs_dyn), pairs_true, pairs_dyn)
    else:
        pairs_dyn, transformations_dyn = torch.tensor([]).reshape(0, 10), torch.tensor([]).reshape(0, 4, 4)
    
    pairs_matched = torch.cat([pairs_sta, pairs_dyn], dim=0)
    transformations_matched = torch.cat([transformations_sta, transformations_dyn], dim=0)
    # assert len(pairs_matched)>0 # likely to be bugs or outliers

    return pairs_matched, transformations_matched

def track(args, point_src, point_dst, label_src, label_dst):
    pairs, transformations = match_pcds(args, point_src, point_dst, label_src, label_dst) 
    # print(f'match_pcds pairs: {pairs}, {transformations}')
    # print(f'match_pcds pairs: {torch.round(pairs[:, 0:2], decimals=2)}')
    return pairs, transformations


def random_choice(m, n):
    assert m>=n
    perm  = torch.randperm(m)
    return perm[0:n]

def pad_segment(seg, max_points):
    padding = seg.new_zeros((max_points, 1)) + 1.0
    if len(seg) > max_points:
        sample_idxs = random_choice(len(seg), max_points)
        seg = seg[sample_idxs, :]
    elif len(seg) < max_points:
        padding[len(seg):] = 0.0
        seg = torch.cat([seg, seg.new_zeros((max_points-len(seg), 3))+1e8], dim=0)
    else: 
        pass
    assert len(seg)==max_points
    return torch.cat([seg, padding], axis=1)

# tok=1 already works decently. topk=5 is for ablation study and works slightly better
def topk_nms(x, k=5, kernel_size=11):
    b, h, w, d = x.shape
    x = x.unsqueeze(1)
    xp = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
    mask = (x == xp).float().clamp(min=0.0)
    xp = x * mask
    votes, idxs = torch.topk(xp.view(b, -1), dim=1, k=k)
    del xp, mask
    return votes, idxs.long()

def transform_points_batch(xyz, pose):
    assert xyz.dim()==3
    assert pose.dim()==3
    assert xyz.shape[2]==4
    assert pose.shape[1]==4
    assert pose.shape[2]==4
    assert len(xyz)==len(pose)
    b, n, _ = xyz.shape
    # print('transform points batch: ', xyz.shape, pose.shape)
    xyzh = torch.cat([xyz[:, :, 0:3], xyz.new_ones((b, n, 1))], dim=-1)
    xyzh_tmp = torch.bmm(xyzh, pose.permute(0, 2, 1))
    return torch.cat([xyzh_tmp[:, :, 0:3], xyz[:, :, -1:]], dim=-1)

def pytorch3d_icp(args, src, dst):
    icp_result = iterative_closest_point(src, dst, 
                                            init_transform=None, 
                                            thres=args.thres_dist,
                                            max_iterations=100,
                                            relative_rmse_thr=1e-6,
                                            estimate_scale=False,
                                            allow_reflection=False,
                                            verbose=False)

    Rs = icp_result.RTs.R
    ts = icp_result.RTs.T

    Rts = torch.cat([Rs, ts[:, None, :]], dim=1) 
    Rts = torch.cat([Rts.permute(0, 2, 1), Rts.new_zeros(len(ts), 1, 4)], dim=1)
    Rts[:, 3, 3]=1.0

    # print('pytorch3d icp Rt: ', Rt)
    # src_tmp = transform_points_tensor(src, Rt)
    # visualize_pcd(np.concatenate([src_tmp.cpu().numpy(), dst.cpu().numpy()], axis=0), 
    #               np.concatenate([np.zeros((len(src)))+1, np.zeros((len(dst)))+2], axis=0), 
    #               num_colors=3,
    #               title=f'registration debug, size: {len(src)} vs {len(dst)}')
    return Rts

def apply_icp(args, src, dst, init_poses):
    src_tmp = transform_points_batch(src, init_poses)

    Rts = pytorch3d_icp(args, src_tmp, dst)
    Rts = torch.bmm(Rts, init_poses)

    # # # pytorch 3d icp might go wrong ! to fix!
    mask_src = src[:, : , -1] > 0.0
    _, error_init = nearest_neighbor_batch(src_tmp, dst)
    error_init = (error_init * mask_src).sum(dim=1) / mask_src.sum(dim=1)

    src_tmp = transform_points_batch(src, Rts)
    _, error_icp = nearest_neighbor_batch(src_tmp, dst)
    error_icp = (error_icp * mask_src).sum(dim=1) / mask_src.sum(dim=1)
    invalid =  error_icp>=error_init
    Rts[invalid] = init_poses[invalid]

    return Rts

from assets.cuda.histlib import histf
def estimate_init_pose_batch(args, src, dst):
    pcd1 = src[:, :, 0:3]
    pcd2 = dst[:, :, 0:3]
    mask1 = src[:, : , -1] > 0.0
    mask2 = dst[:, : , -1] > 0.0

    ###########################################################################################
    eps = 1e-8
    # https://pytorch.org/docs/stable/generated/torch.arange.html#torch-arange
    bins_x = torch.arange(-args.translation_frame, args.translation_frame+args.thres_dist-eps, args.thres_dist)
    bins_y = torch.arange(-args.translation_frame, args.translation_frame+args.thres_dist-eps, args.thres_dist)
    bins_z = torch.arange(-args.thres_dist, args.thres_dist+args.thres_dist-eps, args.thres_dist)
    # print(f'bins: {bins_x.min()} {bins_x.max()} {bins_x.shape}, {bins_z.min()} {bins_z.max()} {bins_z}')

    # bug there: when batch size is large!
    t_hist = histf(dst, src, 
                  bins_x.min(), bins_y.min(), bins_z.min(),
                  bins_x.max(), bins_y.max(), bins_z.max(),
                  len(bins_x), len(bins_y), len(bins_z))
    b, h, w, d = t_hist.shape
    # print(f't_hist.shape: {t_hist.shape} {bins_x.shape} {bins_y.shape} {bins_z.shape} {t_hist.max()}')
    ###########################################################################################

    t_maxs, t_argmaxs = topk_nms(t_hist)
    t_dynamic = torch.stack([ bins_x[t_argmaxs//d//w%h], bins_y[t_argmaxs//d%w], bins_z[t_argmaxs%d] ], dim=-1) + args.thres_dist//2
    # print(f't_dynamic: {t_dynamic.shape}, {t_maxs} {h}, {w}, {d}, {t_dynamic}')
    del t_hist, bins_x, bins_y, bins_z

    n = pcd1.shape[1]
    t_both = torch.cat([t_dynamic, t_dynamic.new_zeros(b, 1, 3)], dim=1)
    k = t_both.shape[1]

    pcd1_ = pcd1[:, None, :, :] + t_both[:, :, None, :]
    pcd2_ = pcd2[:, None, :, :].expand(-1, k, -1, -1)

    _, errors = nearest_neighbor_batch(pcd1_.reshape(b*k, n, 3), pcd2_.reshape(b*k, n, 3))
    _, errors_inv = nearest_neighbor_batch(pcd2_.reshape(b*k, n, 3), pcd1_.reshape(b*k, n, 3))

    # using errors
    errors = (errors.view(b, k, n) * mask1[:, None, :]).sum(dim=-1) / mask1[:, None, :].sum(dim=-1)
    errors_inv = (errors_inv.view(b, k, n) * mask2[:, None, :]).sum(dim=-1) / mask2[:, None, :].sum(dim=-1)
    errors = torch.minimum(errors, errors_inv)
    error, idx = errors.min(dim=-1)
    error_best = error
    t_best = t_both[torch.arange(0, b, device=idx.device), idx, :]
    del pcd1_, pcd2_, errors

    transformation = torch.eye(4)[None].repeat(b, 1, 1)
    transformation[:, 0:3, -1] = t_best
    return transformation

def estimate_init_pose(args, src, dst):
    transformations = []
    assert len(src) == len(dst)
    n = len(src)//args.chunk_size if len(src) % args.chunk_size ==0 else len(src)//args.chunk_size +1
    for k in range(0, n):
        transformation = estimate_init_pose_batch(args, 
                                                  src[k*args.chunk_size: (k+1)*args.chunk_size], 
                                                  dst[k*args.chunk_size: (k+1)*args.chunk_size])
        transformations.append(transformation)
    transformations = torch.vstack(transformations)
    return transformations

def match_eval(args, pcd1, pcd2, transformations):
    pcd1_tmp = transform_points_batch(pcd1, transformations)
    src = pcd1_tmp
    dst = pcd2
    src_mask = pcd1[:, :, -1]>0.0
    dst_mask = pcd2[:, :, -1]>0.0
    src_dst_idxs, src_error = nearest_neighbor_batch(src, dst)
    dst_src_idxs, dst_error = nearest_neighbor_batch(dst, src)

    src_inlier = torch.logical_and(src_error < args.thres_dist, src_mask).float()
    dst_inlier = torch.logical_and(dst_error < args.thres_dist, dst_mask).float()

    src_ratio = torch.sum(src_inlier, dim=1) / torch.sum(src_mask, dim=1)
    dst_ratio = torch.sum(dst_inlier, dim=1) / torch.sum(dst_mask, dim=1)

    src_iou = torch.sum(src_inlier, dim=1) / (torch.sum(src_mask, dim=1) + torch.sum(dst_mask, dim=1) - torch.sum(dst_inlier, dim=1))
    dst_iou = torch.sum(dst_inlier, dim=1) / (torch.sum(src_mask, dim=1) + torch.sum(dst_mask, dim=1) - torch.sum(src_inlier, dim=1))

    src_error = (src_error * src_mask).sum(1) / src_mask.sum(1)
    dst_error = (dst_error * dst_mask).sum(1) / dst_mask.sum(1)

    src_mean = (src[:, :, 0:3] * src_mask[:, :, None]).sum(dim=1) / src_mask.sum(dim=1, keepdim=True)
    src_ori_mean = (pcd1[:, :, 0:3] * src_mask[:, :, None]).sum(dim=1) / src_mask.sum(dim=1, keepdim=True)
    # dst_mean = (dst * dst_mask).sum(dim=1) / dst_mask.sum(dim=1, keepdim=True)
    translations = src_mean - src_ori_mean
    rotations = pytorch3d_t.matrix_to_euler_angles(transformations[:, 0:3, 0:3], convention='ZYX') * 180./np.pi

    return torch.stack([src_error, dst_error], dim=1), \
            torch.stack([src_inlier.sum(1), dst_inlier.sum(1)], dim=1), \
            torch.stack([src_ratio, dst_ratio], dim=1), \
            torch.stack([src_iou, dst_iou], dim=1), \
                translations, rotations

def hist_icp(args, src, dst):
    mask1 = src[:, :, -1]>0.0
    mask2 = dst[:, :, -1]>0.0
    # alwyas match the smaller one to the larger one
    idxs = mask1.sum(dim=1)>mask2.sum(dim=1)
    src_ = src.clone()
    dst_ = dst.clone()
    src_[idxs] = dst[idxs]
    dst_[idxs] = src[idxs]

    with torch.no_grad():
        init_poses_ = estimate_init_pose(args, src_, dst_) 
        transformations_ = apply_icp(args, src_, dst_, init_poses_)

    if sum(idxs)>0:
        transformations = transformations_.clone()
        transformations[idxs] = torch.linalg.inv(transformations_[idxs])
    else:
        transformations = transformations_
    return transformations

def check_transformation(args, translation, rotation, iou):
    # print('check transformation: ', translation, rotation, iou, args.translation_frame)
    # # # check translation
    if torch.linalg.norm(translation) > args.translation_frame:
        return False

    # # # check iou
    if iou<args.thres_iou:
        return  False

    # # # check rotation, in degrees, almost no impact on final result
    max_rot = args.thres_rot * 90.0
    if torch.abs(rotation[1:3]).max()>max_rot: # roll and pitch
        return False

    return True

    
def match_segments_descend(matrix_metric):
    src_idxs = torch.arange(0, len(matrix_metric))
    dst_idxs = torch.argmin(matrix_metric, dim=1)
    return src_idxs, dst_idxs

def match_pairs(args, src_points, dst_points, src_labels, dst_labels, pairs):
    src_labels_unq = torch.unique(src_labels)
    dst_labels_unq = torch.unique(dst_labels)
    matrix_errors = torch.zeros((len(src_labels_unq), len(dst_labels_unq), 2)) + 1e8
    matrix_inliers = torch.zeros((len(src_labels_unq), len(dst_labels_unq), 2)) + 0.0
    matrix_ratios = torch.zeros((len(src_labels_unq), len(dst_labels_unq), 2)) + 0.0
    matrix_ious = torch.zeros((len(src_labels_unq), len(dst_labels_unq), 2)) + 0.0
    matrix_transformations = torch.zeros((len(src_labels_unq), len(dst_labels_unq), 4, 4)) 

    assert len(pairs)>0
    segs_src = []
    segs_dst = []
    for pair in pairs:
        src = src_points[src_labels==pair[0], 0:3]
        dst = dst_points[dst_labels==pair[1], 0:3]
        src = pad_segment(src, args.max_points)
        dst = pad_segment(dst, args.max_points)
        # always match the smaller one to the larger one
        segs_src.append(src)
        segs_dst.append(dst)

    segs_src = torch.stack(segs_src, dim=0)
    segs_dst = torch.stack(segs_dst, dim=0)
    transformations = hist_icp(args, segs_src, segs_dst)
    errors, inliers, ratios, ious, translations, rotations = match_eval(args, segs_src, segs_dst, transformations)
    # reject unreliable matches
    num_matches = 0
    for k, (pair, error, inlier, ratio, iou, translation, rotation, transformation) in \
        enumerate(zip(pairs, errors, inliers, ratios, ious, translations, rotations, transformations)):
        # print('check per pair: ', pair, error, inlier, ratio, iou, translation, rotation, args.translation_max )
        if not check_transformation(args, translation, rotation, min(iou)):
            continue
        src_idx = torch.nonzero(src_labels_unq == pair[0])
        dst_idx = torch.nonzero(dst_labels_unq == pair[1])
        matrix_errors[src_idx, dst_idx, :] = error
        matrix_inliers[src_idx, dst_idx, :] = inlier
        matrix_ratios[src_idx, dst_idx, :] = ratio
        matrix_ious[src_idx, dst_idx, :] = iou
        matrix_transformations[src_idx, dst_idx] = transformation
        num_matches += 1

    if num_matches>0:
        matrix_errors_min, _ = matrix_errors.min(-1)
        src_idxs, dst_idxs = match_segments_descend(matrix_errors_min)
        valid = matrix_errors_min[src_idxs, dst_idxs] < args.thres_error
        src_idxs = src_idxs[valid]
        dst_idxs = dst_idxs[valid]

        # matrix_ious_min, _ = matrix_ious.min(-1)
        # src_idxs, dst_idxs = match_segments_ascend(matrix_ious_min)
        # valid = matrix_ious_min[src_idxs, dst_idxs] >= args.thres_iou
        # src_idxs = src_idxs[valid]
        # dst_idxs = dst_idxs[valid]

        pairs = torch.cat([src_labels_unq[src_idxs][:, None], dst_labels_unq[dst_idxs][:, None], 
                                matrix_errors[src_idxs, dst_idxs], 
                                matrix_inliers[src_idxs, dst_idxs], 
                                matrix_ratios[src_idxs, dst_idxs], 
                                matrix_ious[src_idxs, dst_idxs]], 
                                axis=1)

        transformations = matrix_transformations[src_idxs, dst_idxs]

    else:
        pairs = torch.tensor([]).reshape(0, 2+matrix_errors.shape[-1]+matrix_inliers.shape[-1]+matrix_ratios.shape[-1]+matrix_ious.shape[-1])
        transformations = torch.tensor([]).reshape(0, 4, 4)

    return pairs, transformations


