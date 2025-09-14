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

from .deflow import DeFlow, DeFlowPP
from .fastflow3d import FastFlow3D
from .nsfp import NSFP

# check README for package:
try:
    from .voteflow import VoteFlow
except ImportError as e:
    print("\033[93m--- WARNING [model]: VoteFlow is not imported, as it requires some lib which is not installed.")
    print(f"Detail error message\033[0m: {e}. Just ignore this warning if code runs without these models.")

# following need install extra package: 
# * pip install spconv-cu117
try:
    from .flow4d import Flow4D
except ImportError as e:
    print("\033[93m--- WARNING [model]: Model with SparseConv is not imported, as it requires spconv lib which is not installed.")
    print(f"Detail error message\033[0m: {e}. Just ignore this error if code runs without these models.")

# following need install extra package:
# * pip install torch_scatter mmengine-lite
try:
    from .ssf import SSF
except ImportError as e:
    print("\033[93m--- WARNING [model]: Model with torch scatter is not imported, as it requires some lib which is not installed.")
    print(f"Detail error message\033[0m: {e}. Just ignore this warning if code runs without these models.")

# following need install extra package:
# * pip install FastGeodis --no-build-isolation --no-cache-dir
try:
    from .fastnsf import FastNSF
except ImportError:
    print("--- WARNING [model]: FastNSF is not imported, as it requires FastGeodis lib which is not installed.")

# following need install extra package:
# * pip install pytorch3d assets/cuda/histlib
try:
    from .icpflow import ICPFlow
except ImportError as e:
    print("--- WARNING [model]: ICPFlow is not imported, as it requires pytorch3d lib which is not installed.")
    print(f"Detail error message\033[0m: {e}. Just ignore this warning if code runs without these models.")
