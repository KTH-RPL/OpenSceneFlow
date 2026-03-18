"""
# Created: 2025-08-07 20:12
# Copyright (C) 2025-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of 
# * OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow)
# * HiMo (https://kin-zhang.github.io/HiMo)
#
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
#
"""



from .selfsupervise import *
from .supervise import *

# automatic collection of SSL loss function names for trainer.py
SSL_LOSSES_FN = {name for name in dir(selfsupervise) if name.endswith('Loss') and callable(getattr(selfsupervise, name))}