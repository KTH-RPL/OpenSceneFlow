FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
ENV DEBIAN_FRONTEND noninteractive
LABEL maintainer="Qingwen Zhang <https://kin-zhang.github.io/>"

RUN apt update && apt install -y git tmux curl vim rsync libgl1 libglib2.0-0 ca-certificates

# install zsh and oh-my-zsh
RUN apt update && apt install -y wget git zsh tmux vim g++
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell -p git \
    -p https://github.com/agkozak/zsh-z \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting
    
RUN printf "y\ny\ny\n\n" | bash -c "$(curl -fsSL https://raw.githubusercontent.com/Kin-Zhang/Kin-Zhang/main/scripts/setup_ohmyzsh.sh)"
RUN /opt/conda/bin/conda init zsh

# change to conda env
ENV PATH /opt/conda/bin:$PATH
RUN /opt/conda/bin/conda config --set solver libmamba

RUN mkdir -p /home/kin/workspace && cd /home/kin/workspace && git clone https://github.com/Kin-Zhang/OpenSceneFlow
WORKDIR /home/kin/workspace/OpenSceneFlow

# need read the gpu device info to compile the cuda extension
RUN /opt/conda/bin/pip install -r /home/kin/workspace/OpenSceneFlow/requirements.txt
RUN /opt/conda/bin/pip install FastGeodis --no-build-isolation
RUN /opt/conda/bin/pip install --no-cache-dir -e ./assets/cuda/chamfer3D && /opt/conda/bin/pip install --no-cache-dir -e ./assets/cuda/mmcv

# environment for dataprocessing includes data-api
RUN /opt/conda/bin/conda env create -f envsftool.yaml
RUN /opt/conda/envs/sftool/bin/pip install numpy==1.22

# clean up apt cache
RUN rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip
