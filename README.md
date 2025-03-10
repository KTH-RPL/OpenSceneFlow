<p align="center">
    <picture>
    <img alt="opensceneflow" src="assets/docs/logo.png" width="600">
    </picture><br>
</p>

OpenSceneFlow is a codebase for point cloud scene flow estimation. 
It is also an official implementation of the following paper (sored by the time of publication):

- **Flow4D: Leveraging 4D Voxel Network for LiDAR Scene Flow Estimation**  
*Jaeyeul Kim, Jungwan Woo, Ukcheol Shin, Jean Oh, Sunghoon Im*  
IEEE Robotics and Automation Letters (**RA-L**) 2025  
[ Backbone ] [ Supervised ] - [ [arXiv](https://arxiv.org/abs/2407.07995) ] [ [Project](https://github.com/dgist-cvlab/Flow4D) ] &rarr; [here](#flow4d)

- **SSF: Sparse Long-Range Scene Flow for Autonomous Driving**  
*Ajinkya Khoche, Qingwen Zhang, Laura Pereira Sánchez, Aron Asefaw, Sina Sharif Mansouri and Patric Jensfelt*  
International Conference on Robotics and Automation (**ICRA**) 2025  
[ Backbone ] [ Supervised ] - [ [arXiv](https://arxiv.org/abs/2501.17821) ] [ [Project](https://github.com/KTH-RPL/SSF) ] &rarr; [here](#ssf)

- **SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving**  
*Qingwen Zhang, Yi Yang, Peizheng Li, Olov Andersson, Patric Jensfelt*  
European Conference on Computer Vision (**ECCV**) 2024  
[ Strategy ] [ Self-Supervised ] - [ [arXiv](https://arxiv.org/abs/2407.01702) ] [ [Project](https://github.com/KTH-RPL/SeFlow) ] &rarr; [here](#seflow)

- **DeFlow: Decoder of Scene Flow Network in Autonomous Driving**  
*Qingwen Zhang, Yi Yang, Heng Fang, Ruoyu Geng, Patric Jensfelt*  
International Conference on Robotics and Automation (**ICRA**) 2024  
[ Backbone ] [ Supervised ] - [ [arXiv](https://arxiv.org/abs/2401.16122) ] [ [Project](https://github.com/KTH-RPL/DeFlow) ] &rarr; [here](#deflow)


💞 If you find *OpenSceneFlow* useful to your research, please cite [our works 📖](#cite-us) and give a star 🌟 as encouragement. (੭ˊ꒳​ˋ)੭✧

🎁 <b>One repository, All methods!</b>. Additionally, *OpenSceneFlow* integrates the following excellent work: [ICLR'24 ZeroFlow](https://arxiv.org/abs/2305.10424), [ICCV'23 FastNSF](https://arxiv.org/abs/2304.09121), [RA-L'21 FastFlow](https://arxiv.org/abs/2103.01306), [NeurIPS'21 NSFP](https://arxiv.org/abs/2111.01253), 

<details> <summary> Summary of them:</summary>

- [x] [FastFlow3d](https://arxiv.org/abs/2103.01306): RA-L 2021, a basic backbone model.
- [x] [ZeroFlow](https://arxiv.org/abs/2305.10424): ICLR 2024, their pre-trained weight can covert into our format easily through [the script](tools/zerof2ours.py).
- [ ] [NSFP](https://arxiv.org/abs/2111.01253): NeurIPS 2021, faster 3x than original version because of [our CUDA speed up](assets/cuda/README.md), same (slightly better) performance. Done coding, public after review.
- [ ] [FastNSF](https://arxiv.org/abs/2304.09121): ICCV 2023. Done coding, public after review.
- [ ] [ICP-Flow](https://arxiv.org/abs/2402.17351): CVPR 2024. Done coding, public after review.

</details>

💡: Want to learn how to add your own network in this structure? Check [Contribute section] and know more about the code. Fee free to pull request and your bibtex [here](#cite-us) by pull request.

---

<!-- 📜 Changelog:

- 🎁 2025/1/28 14:58: Update the codebase to collect all methods in one repository reference [Pointcept](https://github.com/Pointcept/Pointcept) repo.
- 🤗 2024/11/18 16:17: Update model and demo data download link through HuggingFace, Personally I found `wget` from HuggingFace link is much faster than Zenodo.
- 2024/09/26 16:24: All codes already uploaded and tested. You can to try training directly by downloading (through [HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow)/[Zenodo](https://zenodo.org/records/13744999)) demo data or pretrained weight for evaluation. 
- 2024/07/24: Merging SeFlow & DeFlow code together, lighter setup and easier running.
- 🔥 2024/07/02: Check the self-supervised version in our new ECCV'24 [SeFlow](https://github.com/KTH-RPL/SeFlow). The 1st ranking in new leaderboard among self-supervise methods. -->

## 0. Installation

**Environment**: Setup

```bash
git clone --recursive https://github.com/KTH-RPL/OpenSceneFlow.git
cd OpenSceneFlow && mamba env create -f environment.yaml
```

CUDA package (need install nvcc compiler), the compile time is around 1-5 minutes:
```bash
mamba activate opensf
# CUDA already install in python environment. I also tested others version like 11.3, 11.4, 11.7, 11.8 all works
cd assets/cuda/mmcv && python ./setup.py install && cd ../../..
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
```

Or you always can choose [Docker](https://en.wikipedia.org/wiki/Docker_(software)) which isolated environment and free yourself from installation, you can pull it by. 
If you have different arch, please build it by yourself `cd OpenSceneFlow && docker build -t zhangkin/opensf` by going through [build-docker-image](assets/README.md#build-docker-image) section.

```bash
# option 1: pull from docker hub
docker pull zhangkin/opensf

# run container
docker run -it --gpus all -v /dev/shm:/dev/shm -v /home/kin/data:/home/kin/data --name opensceneflow zhangkin/opensf /bin/zsh
# and better to read your own gpu device info to compile the cuda extension again:
cd /home/kin/workspace/OpenSceneFlow/assets/cuda/mmcv && /opt/conda/envs/opensf/bin/python ./setup.py install
cd /home/kin/workspace/OpenSceneFlow/assets/cuda/chamfer3D && /opt/conda/envs/opensf/bin/python ./setup.py install
```


## 1. Data Preparation

Refer to [dataprocess/README.md](dataprocess/README.md) for dataset download instructions. Currently, we support **Argoverse 2**, **Waymo**, and **custom datasets** (more datasets will be added in the future). 

After downloading, convert the raw data to `.h5` format for easy training, evaluation, and visualization. Follow the steps in [dataprocess/README.md#process](dataprocess/README.md#process). For a quick start, use our **mini processed dataset**, which includes one scene in `train` and `val`. It is pre-converted to `.h5` format with label data ([Zenodo](https://zenodo.org/records/13744999/files/demo_data.zip)/[HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow/blob/main/demo_data.zip)).


```bash
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/demo_data.zip
unzip demo_data.zip -p /home/kin/data/av2
```

Once extracted, you can directly use this dataset to run the [training script](#2-quick-start) without further processing.

## 2. Quick Start

### Flow4D

Train Flow4D with the leaderboard submit config. [Runtime: Around 18 hours in 4x RTX 3090 GPUs.]

```bash
python train.py model=flow4d lr=1e-3 epochs=15 batch_size=8 num_frames=5 loss_fn=deflowLoss "voxel_size=[0.2, 0.2, 0.2]" "point_cloud_range=[-51.2, -51.2, -3.2, 51.2, 51.2, 3.2]"
```

Pretrained weight can be downloaded through:
```bash
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/flow4d_best.ckpt
```

<!-- ### SSF -->

### SeFlow

Train SeFlow needed to specify the loss function, we set the config of our best model in the leaderboard. [Runtime: Around 11 hours in 4x A100 GPUs.]

```bash
python train.py model=deflow lr=2e-4 epochs=9 batch_size=16 loss_fn=seflowLoss "add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" "model.target.num_iters=2"
```

Pretrained weight can be downloaded through:
```bash
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/seflow_best.ckpt
```

### DeFlow

Train DeFlow with the leaderboard submit config. [Runtime: Around 6-8 hours in 4x A100 GPUs.] Please change `batch_size&lr` accoordingly if you don't have enough GPU memory. (e.g. `batch_size=6` for 24GB GPU)

```bash
python train.py model=deflow lr=2e-4 epochs=15 batch_size=16 loss_fn=deflowLoss
```

Pretrained weight can be downloaded through:
```bash
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/deflow_best.ckpt
```

## 3. Evaluation

You can view Wandb dashboard for the training and evaluation results or upload result to online leaderboard.

Since in training, we save all hyper-parameters and model checkpoints, the only thing you need to do is to specify the checkpoint path. Remember to set the data path correctly also.

```bash
# it will directly prints all metric
python eval.py checkpoint=/home/kin/seflow_best.ckpt av2_mode=val

# it will output the av2_submit.zip or av2_submit_v2.zip for you to submit to leaderboard
python eval.py checkpoint=/home/kin/seflow_best.ckpt av2_mode=test leaderboard_version=1
python eval.py checkpoint=/home/kin/seflow_best.ckpt av2_mode=test leaderboard_version=2
```

To submit to the Online Leaderboard, if you select `av2_mode=test`, it should be a zip file for you to submit to the leaderboard.
Note: The leaderboard result in DeFlow&SeFlow main paper is [version 1](https://eval.ai/web/challenges/challenge-page/2010/evaluation), as [version 2](https://eval.ai/web/challenges/challenge-page/2210/overview) is updated after DeFlow&SeFlow.

```bash
# since the env may conflict we set new on deflow, we directly create new one:
mamba create -n py37 python=3.7
mamba activate py37
pip install "evalai"

# Step 2: login in eval and register your team
evalai set-token <your token>

# Step 3: Copy the command pop above and submit to leaderboard
evalai challenge 2010 phase 4018 submit --file av2_submit.zip --large --private
evalai challenge 2210 phase 4396 submit --file av2_submit_v2.zip --large --private
```

## 4. Visualization

We provide a script to visualize the results of the model also. You can specify the checkpoint path and the data path to visualize the results. The step is quite similar to evaluation.

```bash
python save.py checkpoint=/home/kin/seflow_best.ckpt dataset_path=/home/kin/data/av2/preprocess_v2/sensor/vis

# The output of above command will be like:
Model: DeFlow, Checkpoint from: /home/kin/model_zoo/v2/seflow_best.ckpt
We already write the flow_est into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:
python tools/visualization.py --res_name 'seflow_best' --data_dir /home/kin/data/av2/preprocess_v2/sensor/vis
Enjoy! ^v^ ------ 

# Then run the command in the terminal:
python tools/visualization.py --res_name 'seflow_best' --data_dir /home/kin/data/av2/preprocess_v2/sensor/vis
```

https://github.com/user-attachments/assets/f031d1a2-2d2f-4947-a01f-834ed1c146e6

Or another way to interact with [rerun](https://github.com/rerun-io/rerun) but please only vis scene by scene, not all at once.

```bash
python tools/visualization_rerun.py --data_dir /home/kin/data/av2/h5py/demo/train --res_name "['flow', 'deflow']"
```

https://github.com/user-attachments/assets/07e8d430-a867-42b7-900a-11755949de21


## Cite Us

*OpenSceneFlow* is designed by [Qingwen Zhang](https://kin-zhang.github.io/) from DeFlow and SeFlow project. If you find it useful, please cite our works:

```bibtex
@inproceedings{zhang2024seflow,
  author={Zhang, Qingwen and Yang, Yi and Li, Peizheng and Andersson, Olov and Jensfelt, Patric},
  title={{SeFlow}: A Self-Supervised Scene Flow Method in Autonomous Driving},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024},
  pages={353–369},
  organization={Springer},
  doi={10.1007/978-3-031-73232-4_20},
}
@inproceedings{zhang2024deflow,
  author={Zhang, Qingwen and Yang, Yi and Fang, Heng and Geng, Ruoyu and Jensfelt, Patric},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={{DeFlow}: Decoder of Scene Flow Network in Autonomous Driving}, 
  year={2024},
  pages={2105-2111},
  doi={10.1109/ICRA57147.2024.10610278}
}
```

And our excellent collaborators works as followings:

```bibtex
@article{kim2025flow4d,
  author={Kim, Jaeyeul and Woo, Jungwan and Shin, Ukcheol and Oh, Jean and Im, Sunghoon},
  journal={IEEE Robotics and Automation Letters}, 
  title={Flow4D: Leveraging 4D Voxel Network for LiDAR Scene Flow Estimation}, 
  year={2025},
  volume={10},
  number={4},
  pages={3462-3469},
  doi={10.1109/LRA.2025.3542327}
}
@article{khoche2025ssf,
  title={SSF: Sparse Long-Range Scene Flow for Autonomous Driving},
  author={Khoche, Ajinkya and Zhang, Qingwen and Sanchez, Laura Pereira and Asefaw, Aron and Mansouri, Sina Sharif and Jensfelt, Patric},
  journal={arXiv preprint arXiv:2501.17821},
  year={2025}
}
```

Feel free to contribute your method and add your bibtex here by pull request!

❤️: [BucketedSceneFlowEval](https://github.com/kylevedder/BucketedSceneFlowEval); [Pointcept](https://github.com/Pointcept/Pointcept); [ZeroFlow](https://github.com/kylevedder/zeroflow) ...
