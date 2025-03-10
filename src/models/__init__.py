"""
# Created: 2024-11-21 20:12
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow)
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from .deflow import DeFlow
from .fastflow3d import FastFlow3D

# following need install extra package: 
# * pip install spconv-cu117
try:
    from .flow4d import Flow4D
except ImportError as e:
    print("\033[93m--- WARNING [model]: Model with SparseConv is not imported, as it requires spconv lib which is not installed.")
    print(f"\033[91m--- Detail error message\033[0m: {e}")