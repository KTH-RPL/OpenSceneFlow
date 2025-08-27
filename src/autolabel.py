"""
# Created: 2024-02-22 22:05
# Updated: 2025-08-20 21:45
# Copyright (C) 2024-now, Scania Sverige EEARP Group, KTH Royal Institute of Technology
# Author: Qingwen ZHANG  (https://kin-zhang.github.io/)
# License: GPLv2, allow it free only for academic use.
# 
# Change Logs:
# 2024-02-22: clean up from HiMo project.
# 
# Description: Label strategy for self-supervised learning (SSL) scene flow estimation.
# 
"""

import numpy as np
from copy import deepcopy

def shiftClusterid(cluster):
    """
    Shift the cluster labels by 1:
    0: background, no cluster id
    1: save this label for no cluster_id but dynamic
    2+: cluster id
    """
    shifted_cluster = np.zeros_like(cluster)
    mask = cluster > 0
    shifted_cluster[mask] = cluster[mask] + 1
    return shifted_cluster

# mainly based on dufo with label inside for cluster-loss. Check HiMo Fig. 6 Top
def seflow_auto(input_data):
    dufo = input_data['dufo'][:].astype(np.uint8)
    cluster = shiftClusterid(input_data['dufocluster'][:].astype(np.int16))
    cluster[dufo == 0] = 0
    return cluster

# based on dufo and nnd for dynamic then cluster-wise checking with reassign. Check HiMo Fig. 6 Bottom
def seflowpp_auto(input_data, tau1=0.05, tau2=0.30):
    """
    check HiMo paper (Eq. 5) to know more about paramter setting here.
    We didn't explore this parameter too much feel free to adjust as you want after reading the paper.
    * For highway Scania data we set tau1=0.01, tau2=0.05
    * For urban Argoverse 2 data we set tau1=0.05 tau2=0.3
    """
    dufo = input_data['dufo'][:].astype(np.uint8)
    cluster = shiftClusterid(input_data['cluster'][:].astype(np.int16))
    nnd = input_data['nnd'][:].astype(np.uint8)

    dynamic = np.zeros_like(dufo)
    for cluster_id in np.unique(cluster):
        if cluster_id in [0, 1]:
            continue
        all_pts = np.sum(cluster == cluster_id)
        cluster_dufo = dufo[cluster == cluster_id]
        cluster_nnd = nnd[cluster == cluster_id]
        
        r_dufo = np.sum(cluster_dufo>0)/all_pts
        r_nnd = np.sum(cluster_nnd>0)/all_pts

        if min(r_dufo, r_nnd) > tau1 and max(r_dufo, r_nnd) > tau2:
            dynamic[cluster == cluster_id] = cluster_id
    return dynamic
