OpenSceneFlow Assets
---

There are two ways to setup the environment: conda in your desktop and docker container isolate environment.

## Docker Environment

### Build Docker Image
If you want to build docker with compile all things inside, there are some things need setup first in your own desktop environment: 
- [NVIDIA-driver](https://www.nvidia.com/download/index.aspx): which I believe most of people may already have it. Try `nvidia-smi` to check if you have it.
- [Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository):
   ```bash
   # Add Docker's official GPG key:
   sudo apt-get update
   sudo apt-get install ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   sudo chmod a+r /etc/apt/keyrings/docker.asc

   # Add the repository to Apt sources:
   echo \
   "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
   $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
   sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   ```
- [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
   ```bash
   sudo apt update && apt install nvidia-container-toolkit
   ```

Then follow [this stackoverflow answers](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime):
1. Edit/create the /etc/docker/daemon.json with content:
   ```bash
   {
      "runtimes": {
         "nvidia": {
               "path": "/usr/bin/nvidia-container-runtime",
               "runtimeArgs": []
            } 
      },
      "default-runtime": "nvidia" 
   }
   ```
2. Restart docker daemon:
   ```bash
   sudo systemctl restart docker
   ```

3. Then you can build the docker image:
   ```bash
   cd OpenSceneFlow && docker build -f Dockerfile -t zhangkin/opensf .
   ```

### To Apptainer container

If you want to build a **minimal** training env for Apptainer container, you can use the following command:
```bash
apptainer build opensf.sif assets/opensf.def
# zhangkin/opensf:full is created by Dockerfile
```

Then run as a Python env with:
```bash
PYTHON="apptainer run --nv --writable-tmpfs opensf.sif"
$PYTHON train.py
```

<!-- 
In case the compile package not working for your CUDA cability, add following code to the `assets/opensf.def` file before `exec`:
```bash
echo "Running pip install for local CUDA modules..."
/opt/conda/bin/pip install /workspace/assets/cuda/chamfer3D
/opt/conda/bin/pip install /workspace/assets/cuda/mmcv
``` -->


## Installation

We will use conda to manage the environment with mamba for faster package installation.

### System
Install conda with mamba for package management and  for faster package installation:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

### Environment

Create base env (Python 3.10 + torch 2.6.0 + CUDA 11.8) and compile our CUDA extensions: [around 5 minutes based on your network speed and cpu]

```bash
git clone https://github.com/KTH-RPL/OpenSceneFlow.git
cd OpenSceneFlow
mamba env create -f environment.yaml
mamba activate opensf
pip install --no-build-isolation --no-deps ./assets/cuda/chamfer3D ./assets/cuda/mmcv
```

The extra packages that only some methods need (FastGeodis, pytorch3d, histlib, cpp_im2ht) are listed in [the main README](../README.md#environment-setup).

Checking important packages in our environment now:
```bash
mamba activate opensf
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
python -c "import lightning.pytorch as pl; print('pl version:', pl.__version__)"
python -c "import spconv.pytorch as spconv; print('spconv import successfully')"
python -c "from assets.cuda.mmcv import Voxelization, DynamicScatter;print('successfully import on our lite mmcv package')"
python -c "from assets.cuda.chamfer3D import nnChamferDis;print('successfully import on our chamfer3D package')"
python -c "from av2.utils.io import read_feather; print('av2 package ok') "
```


### Other issues

<!-- 1. looks like open3d and fire package conflict, not sure
   -  need install package like `pip install --ignore-installed`, ref: [pip cannot install distutils installed project](https://stackoverflow.com/questions/53807511/pip-cannot-uninstall-package-it-is-a-distutils-installed-project), my error: `ERROR: Cannot uninstall 'blinker'.`
   -  need specific werkzeug version for open3d 0.16.0, otherwise error: `ImportError: cannot import name 'url_quote' from 'werkzeug.urls'`. But need update to solve the problem: `pip install --upgrade Flask` [ref](https://stackoverflow.com/questions/77213053/why-did-flask-start-failing-with-importerror-cannot-import-name-url-quote-fr) -->


1. `ModuleNotFoundError: No module named 'torch'` when pip compiles one of our CUDA extensions (or FastGeodis, pytorch3d):
   their `setup.py` imports torch, but pip builds a package in an empty isolated env by default. Install with `pip install --no-build-isolation`, as in the commands above.

2. A wrong package version is imported (numpy 2, another torch, ...) although the env is correct:
   Python also reads the packages in `~/.local/lib/python3.10/site-packages` and in `PYTHONPATH` (ROS sets it). Run with `PYTHONNOUSERSITE=1` and an empty `PYTHONPATH`, also when you create the env.

3. `nvcc -V` is not 11.8, or thrust headers do not match when compiling:
   `nvidia/label/cuda-11.8.0` must be the first channel in `environment.yaml`, before conda-forge. Otherwise the parts of the CUDA toolkit that are also on conda-forge are installed in their newest 12.x version.

4. In cluster have error: `pandas ImportError: /lib64/libstdc++.so.6: version 'GLIBCXX_3.4.29' not found`
    Solved by `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/proj/berzelius-2023-154/users/x_qinzh/mambaforge/lib`

5. cuda package problem: `ValueError(f"Unknown CUDA arch ({arch}) or GPU not supported")`
   `environment.yaml` sets `TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"` when the env is activated (RTX 20/30/40 series, A100, H100). For another GPU, [check its compute capability](https://developer.nvidia.cn/cuda-gpus#compute) and set it yourself before compiling: `export TORCH_CUDA_ARCH_LIST=8.6`

6. torch-scatter: `OSError: ... version 'GLIBC_2.32' not found`
   The torch-scatter wheels for torch 2.7 need GLIBC 2.32 (Ubuntu 20.04 has 2.31), that is why `environment.yaml` has torch 2.6.0.

7. `ImportError: libgthread-2.0.so.0: cannot open shared object file` (or `libGL.so.1`) when importing cv2, on a server or in a container:
   opencv-python, which av2 installs, needs two system libraries that a desktop already has: `sudo apt install libglib2.0-0 libgl1`

Issues of the previous environment (Python 3.8 + torch 2.0.0 + CUDA 11.7):

1. `ImportError: libtorch_cuda.so: undefined symbol: cudaGraphInstantiateWithFlags, version libcudart.so.11.0`
   The cuda version: `pytorch::pytorch-cuda` and `nvidia::cudatoolkit` need be same. [Reference link](https://github.com/pytorch/pytorch/issues/90673#issuecomment-1563799299)

2. torch_scatter problem: `OSError: /home/kin/mambaforge/envs/opensf-v2/lib/python3.10/site-packages/torch_scatter/_version_cpu.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE`
   Solved by install the torch-cuda version: `pip install https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.2%2Bpt20cu118-cp310-cp310-linux_x86_64.whl`
