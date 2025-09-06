"""
# Created: 2023-12-18 17:23
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
#
#
# Description: Test which existing hdbscan speed
"""
import os, sys
BASEF_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '../..'))
sys.path.append(BASEF_DIR)
import torch, time
import numpy as np

# ------ Three different hdbscan implementations ------
from cuml.cluster import hdbscan
from sklearn.cluster import HDBSCAN
import hdbscan as cpu_hdbscan

from src.utils.o3d_view import MyVisualizer, color_map
import open3d as o3d

MAX_AXIS_RANGE = 60

def vis(pc0, labels, title='HDBSCAN'):
    # visualize
    vis = MyVisualizer(view_file=f'{BASEF_DIR}/assets/view/demo.json', window_title=title)
    pcd = o3d.geometry.PointCloud()
    num_points = pc0.shape[0]
    pcd.points = o3d.utility.Vector3dVector(pc0)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((num_points, 3)))
    for i in range(num_points):
        if labels[i] >= 0:
            # print(i, labels[i])
            pcd.colors[i] = color_map[labels[i] % len(color_map)]
    vis.show([pcd])

if __name__ == "__main__":
    pc0 = np.load(f'{BASEF_DIR}/assets/tests/test_pc0.npy')
    pc1 = np.load(f'{BASEF_DIR}/assets/tests/test_pc1.npy')
    print('0: {:.3f}MB'.format(torch.cuda.memory_allocated()/1024**2))
    pc0 = torch.from_numpy(pc0[...,:3]).float().cuda().contiguous()
    pc1 = torch.from_numpy(pc1[...,:3]).float().cuda().contiguous()

    # filter out MAX_AXIS_RANGE
    pc0 = pc0[torch.logical_and(torch.abs(pc0[:,0])<MAX_AXIS_RANGE, torch.abs(pc0[:,1])<MAX_AXIS_RANGE)]
    pc1 = pc1[torch.logical_and(torch.abs(pc1[:,0])<MAX_AXIS_RANGE, torch.abs(pc1[:,1])<MAX_AXIS_RANGE)]
    print(pc0.shape, "demo data: ", pc0[0])
    print(pc1.shape, "demo data: ", pc1[0])
    print('1: {:.3f}MB'.format(torch.cuda.memory_allocated()/1024**2))

    print("------ START GPU cuml HDBSCAN Clustering ------")
    start_t = time.time()
    hdb = hdbscan.HDBSCAN(min_cluster_size=20, cluster_selection_epsilon=0.7)
    for pc in [pc0, pc1]:
        hdb.fit(pc)
        labels = hdb.labels_
    labels = np.array(hdb.labels_.get())
    print(f'cuml hdbscan time cost: {time.time()-start_t:.3f}s')

    pc0 = pc0.cpu().numpy()
    pc1 = pc1.cpu().numpy()
    # vis(pc1, labels, title='cuml HDBSCAN')

    print("------ START CPU sklearn HDBSCAN ------")
    start_t = time.time()
    hdb = HDBSCAN(min_cluster_size=20, cluster_selection_epsilon=0.7, n_jobs=-1)
    for pc in [pc0, pc1]:
        hdb.fit(pc)
        labels = hdb.labels_
    print(f'sklearn hdbscan time cost: {time.time()-start_t:.3f}s')
    # vis(pc1, labels, title='sklearn HDBSCAN')

    print("------ START CPU scikit-learn HDBSCAN ------")
    start_t = time.time()
    cluster = cpu_hdbscan.HDBSCAN(min_cluster_size=20, cluster_selection_epsilon=0.7)
    for pc in [pc0, pc1]:
        cluster_labels = cluster.fit_predict(pc)
    print(f'scikit-learn hdbscan time cost: {time.time()-start_t:.3f}s')
    # vis(pc1, cluster_labels, title='scikit-learn HDBSCAN')
        
"""
Output in my desktop with a 3090 GPU:
python assets/tests/hdbscan_speed.py
----------------------------------------

0: 0.000MB
torch.Size([85396, 3]) demo data:  tensor([-8.2266,  8.3516,  1.4922], device='cuda:0')
torch.Size([85380, 3]) demo data:  tensor([-7.9961,  8.1328,  0.4839], device='cuda:0')
1: 1.955MB
------ START GPU cuml HDBSCAN Clustering ------
cuml hdbscan time cost: 1.973s
------ START CPU sklearn HDBSCAN ------
sklearn hdbscan time cost: 31.773s
------ START CPU scikit-learn HDBSCAN ------
scikit-learn hdbscan time cost: 3.188s

"""