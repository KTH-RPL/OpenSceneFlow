# check more: https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt install -y git curl vim rsync htop

RUN curl -o ~/miniforge3.sh -LO https://github.com/conda-forge/miniforge/releases/latest/download/miniforge3-Linux-x86_64.sh  && \
    chmod +x ~/miniforge3.sh && \
    ~/miniforge3.sh -b -p /opt/conda && \
    rm ~/miniforge3.sh && \
    /opt/conda/bin/conda clean -ya && /opt/conda/bin/conda init bash

# install zsh and oh-my-zsh
RUN apt update && apt install -y wget git zsh tmux vim g++
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell -p git \
    -p https://github.com/agkozak/zsh-z \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting
    
RUN printf "y\ny\ny\n\n" | bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kin-Zhang/Kin-Zhang/main/scripts/setup_ohmyzsh.sh)"
RUN /opt/conda/bin/conda init zsh && /opt/conda/bin/mamba init zsh

# change to conda env
ENV PATH /opt/conda/bin:$PATH

RUN mkdir -p /home/kin/workspace && cd /home/kin/workspace && git clone https://github.com/KTH-RPL/OpenSceneFlow.git
WORKDIR /home/kin/workspace/OpenSceneFlow
RUN apt-get update && apt-get install libgl1 -y
# need read the gpu device info to compile the cuda extension
RUN cd /home/kin/workspace/OpenSceneFlow && /opt/conda/bin/mamba env create -f environment.yaml
RUN cd /home/kin/workspace/OpenSceneFlow/assets/cuda/mmcv && /opt/conda/envs/opensf/bin/python ./setup.py install
RUN cd /home/kin/workspace/OpenSceneFlow/assets/cuda/chamfer3D && /opt/conda/envs/opensf/bin/python ./setup.py install

