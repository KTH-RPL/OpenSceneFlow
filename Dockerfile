# check more: https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
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
RUN apt-get update && apt-get install libgl1 -y

# need read the gpu device info to compile the cuda extension
RUN cd /home/kin/workspace/OpenSceneFlow && /opt/conda/bin/conda env create -f environment.yaml
# To make images can run all methods in the codebase
RUN /opt/conda/envs/opensf/bin/pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
RUN /opt/conda/envs/opensf/bin/pip install FastGeodis --no-build-isolation --no-cache-dir
RUN /opt/conda/envs/opensf/bin/pip install mmengine-lite && \
    /opt/conda/bin/conda install -n opensf -y pytorch3d -c pytorch3d

# custom cuda library
RUN cd /home/kin/workspace/OpenSceneFlow/assets/cuda/mmcv && /opt/conda/envs/opensf/bin/python ./setup.py install
RUN cd /home/kin/workspace/OpenSceneFlow/assets/cuda/chamfer3D && /opt/conda/envs/opensf/bin/python ./setup.py install

RUN cd /home/kin/workspace/OpenSceneFlow && /opt/conda/bin/conda env create -f envsftool.yaml
RUN /opt/conda/envs/sftool/bin/pip install numpy==1.22

# clean up apt cache
RUN rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip && /opt/conda/bin/conda clean -ya
