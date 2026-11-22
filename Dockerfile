# check more: https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
LABEL maintainer="Qingwen Zhang <https://kin-zhang.github.io/>"

RUN apt update && apt install -y git curl vim rsync htop

RUN curl -o ~/miniforge3.sh -LO https://github.com/conda-forge/miniforge/releases/latest/download/miniforge3-Linux-x86_64.sh  && \
    chmod +x ~/miniforge3.sh && \
    ~/miniforge3.sh -b -p /opt/conda && \
    rm ~/miniforge3.sh && \
    /opt/conda/bin/conda clean -ya && /opt/conda/bin/conda init bash && /opt/conda/bin/conda init zsh

# install zsh and oh-my-zsh
RUN apt update && apt install -y wget git zsh tmux vim g++
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell -p git \
    -p https://github.com/agkozak/zsh-z \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting
    
RUN printf "y\ny\ny\n\n" | bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kin-Zhang/Kin-Zhang/main/scripts/setup_ohmyzsh.sh)"

# change to conda env
ENV PATH /opt/conda/bin:$PATH

RUN mkdir -p /home/kin/workspace && cd /home/kin/workspace && git clone https://github.com/KTH-RPL/OpenSceneFlow.git
WORKDIR /home/kin/workspace/OpenSceneFlow
# libglib2.0-0: opencv-python (from av2) needs libgthread-2.0.so.0
RUN apt-get update && apt-get install libgl1 libglib2.0-0 -y

# python 3.10, torch 2.6.0, CUDA 11.8. The compilers (gcc 11, nvcc 11.8) come with the conda env.
RUN cd /home/kin/workspace/OpenSceneFlow && /opt/conda/bin/conda env create -f environment.yaml

# Below are the same steps as in the installation section of README.md. `conda run` activates the env, so that
# TORCH_CUDA_ARCH_LIST of environment.yaml is set: the extensions are compiled for every supported GPU,
# not only for the GPU of the machine that builds the image.
# To make images can run all methods in the codebase: FastGeodis (FastNSF, Floxels), pytorch3d (VoteFlow, ICP-Flow)
RUN /opt/conda/bin/conda run -n opensf --no-capture-output pip install --no-build-isolation --no-cache-dir FastGeodis==1.0.5
RUN /opt/conda/bin/conda run -n opensf --no-capture-output pip install --no-build-isolation --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"

# custom cuda library: chamfer3D and mmcv (all methods), histlib (ICP-Flow), cpp_im2ht (VoteFlow)
RUN /opt/conda/bin/conda run -n opensf --no-capture-output pip install --no-build-isolation --no-deps \
    ./assets/cuda/chamfer3D ./assets/cuda/mmcv ./assets/cuda/histlib ./src/models/basic/voteflow_plugin/hough_transformation/cpp_im2ht

RUN cd /home/kin/workspace/OpenSceneFlow && /opt/conda/bin/conda env create -f envsftool.yaml
RUN /opt/conda/envs/sftool/bin/pip install numpy==1.22

# clean up apt cache
RUN rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip && /opt/conda/bin/conda clean -ya
