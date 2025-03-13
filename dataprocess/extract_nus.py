"""
# 
# Created: 2024-02-24 10:48
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen ZHANG  (https://kin-zhang.github.io/), Ajinkya Khoche, Peizheng Li
#
# Description: Preprocess Data, save as h5df format for faster loading
# This one is for nuScenes dataset
# 
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing
from pathlib import Path
from multiprocessing import Pool, current_process
from typing import Optional, Tuple, Dict, Union, Final
from tqdm import tqdm
import numpy as np
import fire, time, h5py

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion

import os, sys
PARENT_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..'))
sys.path.append(PARENT_DIR)
from src.utils.mics import create_reading_index
from src.utils.av2_eval import CATEGORY_TO_INDEX, NusNamMap
from linefit import ground_seg

BOUNDING_BOX_EXPANSION: Final = 0.2
GROUNDSEG_config = f"{PARENT_DIR}/conf/others/nuscenes.toml"