class bc:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ====> import func through string, ref: https://stackoverflow.com/a/19393328
import importlib
def import_func(path: str):
    function_string = path
    mod_name, func_name = function_string.rsplit('.',1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func

import numpy as np
def npcal_pose0to1(pose0, pose1):
    """
    Note(Qingwen 2023-12-05 11:09):
    Don't know why but it needed set the pose to float64 to calculate the inverse 
    otherwise it will be not expected result....
    """
    pose1_inv = np.eye(4, dtype=np.float64)
    pose1_inv[:3,:3] = pose1[:3,:3].T
    pose1_inv[:3,3] = (pose1[:3,:3].T * -pose1[:3,3]).sum(axis=1)
    pose_0to1 = pose1_inv @ pose0.astype(np.float64)
    return pose_0to1.astype(np.float32)


# a quick inline tee class to log stdout to file
import sys
import re
from datetime import datetime
class InlineTee:
    def __init__(self, filepath, append=False, timestamp_per_line=False):
        mode = "a" if append else "w"
        self.file = open(filepath, mode)
        self.stdout = sys.stdout
        self.newline = True
        self.timestamp_per_line = timestamp_per_line
        self.first_write = True
        self.ansi_pattern = re.compile(r'\x1b\[[0-9;]*m')
        
        # write header timestamp
        if not self.timestamp_per_line:
            self.file.write(f"=== Log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

    def write(self, data):
        self.stdout.write(data)
        
        clean_data = self.ansi_pattern.sub('', data)
        
        if self.timestamp_per_line and clean_data.strip():
            lines = clean_data.split('\n')
            for i, line in enumerate(lines):
                if line and self.newline:
                    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
                    self.file.write(timestamp + line)
                else:
                    self.file.write(line)
                if i < len(lines) - 1:
                    self.file.write('\n')
                    self.newline = True
                else:
                    self.newline = line == ''
        else:
            self.file.write(clean_data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()