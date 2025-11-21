<p align="center">
    <a href="https://github.com/KTH-RPL/OpenSceneFlow">
    <picture>
    <img alt="opensceneflow" src="assets/docs/logo.png" width="600">
    </picture><br>
    </a>
</p>

💞 If you find [*OpenSceneFlow*](https://github.com/KTH-RPL/OpenSceneFlow) useful to your research, please cite [**our works** 📖](#cite-us) and [give a star 🌟](https://github.com/KTH-RPL/OpenSceneFlow) as encouragement. (੭ˊ꒳​ˋ)੭✧

OpenSceneFlow is a codebase for point cloud scene flow estimation. 
It is also an official implementation of the following papers (sorted by the time of publication):

<!-- - **TeFlow: An Efficient Multi-frame Scene Flow Estimation Method**   
*Qingwen Zhang, Chenhan Jiang, Xiaomeng Zhu, Yunqi Miao, Yushan Zhang, Olov Andersson, Patric Jensfelt*  
Under Review   
[ Strategy ] [ Self-Supervised ] - [ [OpenReview](https://openreview.net/forum?id=h70FLgnIAw) ] [ [Project](https://github.com/Kin-Zhang/TeFlow) ]&rarr; [here](#teflow) -->

- **DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method**   
*Qingwen Zhang, Xiaomeng Zhu, Yushan Zhang, Yixi Cai, Olov Andersson, Patric Jensfelt*  
Conference on Neural Information Processing Systems (**NeurIPS**) 2025 - Spotlight   
[ Backbone ] [ Supervised ] - [ [arXiv](https://arxiv.org/abs/2508.17054) ] [ [Project](https://github.com/Kin-Zhang/DeltaFlow) ]&rarr; [here](#deltaflow)

- **HiMo: High-Speed Objects Motion Compensation in Point Clouds** (SeFlow++)   
*Qingwen Zhang, Ajinkya Khoche, Yi Yang, Li Ling, Sina Sharif Mansouri, Olov Andersson, Patric Jensfelt*  
IEEE Transactions on Robotics (**T-RO**) 2025   
[ Strategy ] [ Self-Supervised ] - [ [arXiv](https://arxiv.org/abs/2503.00803) ] [ [Project](https://kin-zhang.github.io/HiMo/) ] &rarr; [here](#seflow-1)

- **VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow**   
*Yancong Lin\*, Shiming Wang\*, Liangliang Nan, Julian Kooij, Holger Caesar*   
Conference on Computer Vision and Pattern Recognition (**CVPR**) 2025  
[ Backbone ] [ Self-Supervised ] - [ [arXiv](https://arxiv.org/abs/2503.22328) ] [ [Project](https://github.com/tudelft-iv/VoteFlow/) ] &rarr; [here](#VoteFLow)

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

🎁 <b>One repository, All methods!</b> 
Additionally, *OpenSceneFlow* integrates following excellent works: [ICLR'24 ZeroFlow](https://arxiv.org/abs/2305.10424), [CVPR'24 ICP-Flow](https://arxiv.org/abs/2402.17351), [ICCV'23 FastNSF](https://arxiv.org/abs/2304.09121), [RA-L'21 FastFlow3D](https://arxiv.org/abs/2103.01306), [NeurIPS'21 NSFP](https://arxiv.org/abs/2111.01253). (More on the way...)

<details> <summary> Summary of them:</summary>

- [x] [FastFlow3D](https://arxiv.org/abs/2103.01306): RA-L 2021, a basic backbone model.
- [x] [ZeroFlow](https://arxiv.org/abs/2305.10424): ICLR 2024, their pre-trained weight can covert into our format easily through [the script](tools/zerof2ours.py).
- [x] [NSFP](https://arxiv.org/abs/2111.01253): NeurIPS 2021, faster 3x than original version because of [our CUDA speed up](assets/cuda/README.md), same (slightly better) performance.
- [x] [FastNSF](https://arxiv.org/abs/2304.09121): ICCV 2023. SSL Optimization-based.
- [x] [ICP-Flow](https://arxiv.org/abs/2402.17351): CVPR 2024. SSL Optimization-based.
- [ ] [EulerFlow](https://arxiv.org/abs/2410.02031): ICLR 2025. SSL optimization-based. In my plan, haven't coding yet.

</details>

💡: Want to learn how to add your own network in this structure? Check [Contribute section](CONTRIBUTING.md#adding-a-new-method) and know more about the code. Fee free to pull request and your bibtex [here](#cite-us).

## 0. Installation

There are two ways to install the codebase: directly on your [local machine](#environment-setup) or in a [Docker container](#docker-recommended-for-isolation).

### Environment Setup

We use conda to manage the environment, you can install it follow [here](assets/README.md#system). Then create the base environment with the following command [5~15 minutes]:

```bash
git clone --recursive https://github.com/KTH-RPL/OpenSceneFlow.git
cd OpenSceneFlow && conda env create -f environment.yaml

# You may need export your LD_LIBRARY_PATH with env lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kin/mambaforge/lib
```
We also provide [requirements.txt](requirements.txt), please check usage through [Dockerfile](Dockerfile).

### Docker (Recommended for Isolation)

You always can choose [Docker](https://en.wikipedia.org/wiki/Docker_(software)) which isolated environment and free yourself from installation. Pull the pre-built Docker image or build manually.

```bash
# option 1: pull from docker hub
docker pull zhangkin/opensf

# run container
docker run -it --net=host --gpus all -v /dev/shm:/dev/shm -v /home/kin/data:/home/kin/data --name opensf zhangkin/opensf /bin/zsh

# and better to read your own gpu device info to compile the cuda extension again:
cd /home/kin/workspace/OpenSceneFlow && git pull
cd /home/kin/workspace/OpenSceneFlow/assets/cuda/mmcv && /opt/conda/envs/opensf/bin/python ./setup.py install
cd /home/kin/workspace/OpenSceneFlow/assets/cuda/chamfer3D && /opt/conda/envs/opensf/bin/python ./setup.py install
cd /home/kin/workspace/OpenSceneFlow
conda activate opensf
```

If you prefer to build the Docker image by yourself, Check [build-docker-image](assets/README.md#build-docker-image) section for more details.

## 1. Data Preparation

Refer to [dataprocess/README.md](dataprocess/README.md) for dataset download instructions. Currently, we support [**Argoverse 2**](https://www.argoverse.org/av2.html), [**Waymo**](https://waymo.com/open/), [**nuScenes**](https://www.nuscenes.org/), [**MAN-TruckScene**](https://github.com/TUMFTM/truckscenes-devkit), [**ZOD**](https://github.com/zenseact/zod) and **custom datasets** (more datasets will be added in the future). 

After downloading, convert the raw data to `.h5` format for easy training, evaluation, and visualization. Follow the steps in [dataprocess/README.md#process](dataprocess/README.md#process). 

For a quick start, use our **mini processed dataset**, which includes one scene in `train` and `val`. It is pre-converted to `.h5` format with label data ([HuggingFace](https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/demo-data-v2.zip)).


```bash
# around 1.3G
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/demo-data-v2.zip
unzip demo-data-v2.zip -d /home/kin/data/av2/h5py
```

Once extracted, you can directly use this dataset to run the [training script](#2-quick-start) without further processing.

## 2. Quick Start

Some tips before running the code:
* Don't forget to active Python environment before running the code. 
* If you want to use [wandb](wandb.ai), replace all `entity="kth-rpl",` to your own entity otherwise tensorboard will be used locally.
* Set correct data path by passing the config, e.g. `train_data=/home/kin/data/av2/h5py/demo/train val_data=/home/kin/data/av2/h5py/demo/val`.

And free yourself from trainning, you can download the pretrained weight from [**HuggingFace - OpenSceneFlow**](https://huggingface.co/kin-zhang/OpenSceneFlow) and we provided the detail `wget` command in each model section. For optimization-based method, it's train-free so you can directly run with [3. Evaluation](#3-evaluation) (check more in the evaluation section).

```bash
conda activate opensf
```

### Supervised Training

#### DeltaFlow

Train DeltaFlow with the leaderboard submit config. [Runtime: Around 18 hours in 10x RTX 3080 GPUs.]

```bash
# total bz then it's 10x2 under above training setup.
python train.py model=deltaFlow optimizer.lr=2e-3 epochs=20 batch_size=2 num_frames=5 loss_fn=deflowLoss train_aug=True "voxel_size=[0.15, 0.15, 0.15]" "point_cloud_range=[-38.4, -38.4, -3.2, 38.4, 38.4, 3.2]" +optimizer.scheduler.name=WarmupCosLR +optimizer.scheduler.max_lr=2e-3 +optimizer.scheduler.total_steps=20000

# Pretrained weight can be downloaded through (av2), check all other datasets in the same folder.
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/deltaflow/deltaflow-av2.ckpt
```

#### Flow4D

Train Flow4D with the leaderboard submit config. [Runtime: Around 18 hours in 4x RTX 3090 GPUs.]

```bash
python train.py model=flow4d optimizer.lr=1e-3 epochs=15 batch_size=8 num_frames=5 loss_fn=deflowLoss "voxel_size=[0.2, 0.2, 0.2]" "point_cloud_range=[-51.2, -51.2, -3.2, 51.2, 51.2, 3.2]"

# Pretrained weight can be downloaded through:
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/flow4d_best.ckpt
```

#### SSF

Extra packages needed for SSF model:
```bash
pip install mmengine-lite
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

Train SSF with the leaderboard submit config. [Runtime: Around 6 hours in 8x A100 GPUs.]

```bash
python train.py model=ssf optimizer.lr=8e-3 epochs=25 batch_size=64 loss_fn=deflowLoss "voxel_size=[0.2, 0.2, 6]" "point_cloud_range=[-51.2, -51.2, -3, 51.2, 51.2, 3]"
```

Pretrained weight can be downloaded through:
```bash
# the leaderboard weight
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/ssf_best.ckpt

# the long-range weight:
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/ssf_long.ckpt
```

#### DeFlow

Train DeFlow with the leaderboard submit config. [Runtime: Around 6-8 hours in 4x A100 GPUs.] Please change `batch_size&lr` accoordingly if you don't have enough GPU memory. (e.g. `batch_size=6` for 24GB GPU)

```bash
python train.py model=deflow optimizer.lr=2e-4 epochs=15 batch_size=16 loss_fn=deflowLoss

# Pretrained weight can be downloaded through:
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/deflow_best.ckpt
```

### Feed-Forward Self-Supervised Model Training

Train Feed-forward SSL methods (e.g. SeFlow/SeFlow++/VoteFlow etc), we needed to:
1) process auto-label process for training. Check [dataprocess/README.md#self-supervised-process](dataprocess/README.md#self-supervised-process) for more details. We provide these inside the demo dataset already.
2) specify the loss function, we set the config here for our best model in the leaderboard.

#### SeFlow

```bash
# [Runtime: Around 11 hours in 4x A100 GPUs.]
python train.py model=deflow optimizer.lr=2e-4 epochs=9 batch_size=16 loss_fn=seflowLoss +ssl_label=seflow_auto "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" "model.target.num_iters=2"

# Pretrained weight can be downloaded through:
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/seflow_best.ckpt
```

#### VoteFLow
Extra pakcges needed for VoteFlow, [pytorch3d](https://pytorch3d.org/) (prefer 0.7.7) and [torch-scatter](https://github.com/rusty1s/pytorch_scatter?tab=readme-ov-file) (prefer 2.1.2):

```bash
# Install Pytorch3d
conda install pytorch3d -c pytorch3d

# Install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

Train VoteFlow with the leaderboard submit config. [Runtime: Around 32 hours in 4 x V100 GPUs.]
```bash
python train.py model=voteflow optimizer.lr=2e-4 +optimizer.scheduler.name=StepLR +optimizer.scheduler.step_size=6 epochs=12 batch_size=4 model.target.m=8 model.target.n=128 loss_fn=seflowLoss "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" +ssl_label=seflow_auto

# Pretrained weight can be downloaded through:
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/voteflow_best.ckpt
```


#### SeFlow++

```bash
# [Runtime: Around 10 hours in 4x A100 GPUs.] for Argoverse 2
python train.py model=deflowpp save_top_model=3 val_every=3 voxel_size="[0.2, 0.2, 6]" point_cloud_range="[-51.2, -51.2, -3, 51.2, 51.2, 3]" num_workers=16 epochs=9 optimizer.lr=2e-4 +optimizer.scheduler.name=StepLR "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" +ssl_label=seflowpp_auto loss_fn=seflowppLoss num_frames=3 batch_size=4

# Pretrained weight can be downloaded through:
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/seflowpp_best.ckpt
```


### Optimization-based Unsupervised Methods

For all optimization-based methods, you can directly run `eval.py`/`save.py` to get the result without training, while the running might take really long time, maybe tmux for run it. For multi-program running, the master port can be set through `+master_port=12346`.

```bash
# you can change another model by passing model name.
python eval.py model=fastnsf

# or save the result directly
python save.py model=fastnsf
```


## 3. Evaluation

You can view Wandb dashboard for the training and evaluation results or upload result to online leaderboard. 
<!-- Three-way EPE and Dynamic Bucket-normalized are evaluated within a 70x70m range (followed Argoverse 2 online leaderboard). No ground points are considered in the evaluation. -->

Since in training, we save all hyper-parameters and model checkpoints, the only thing you need to do is to specify the checkpoint path. Remember to set the data path correctly also.

```bash
# (feed-forward): load ckpt and run it, it will directly prints all metric
python eval.py checkpoint=/home/kin/seflow_best.ckpt data_mode=val

# (optimization-based): it might need take really long time, maybe tmux for run it.
python eval.py model=nsfp +master_port=12344 # change diff port if you want to have multiple runners.

# it will output the av2_submit.zip or av2_submit_v2.zip for you to submit to leaderboard
python eval.py checkpoint=/home/kin/seflow_best.ckpt data_mode=test leaderboard_version=1
python eval.py checkpoint=/home/kin/seflow_best.ckpt data_mode=test leaderboard_version=2
```

### **📊 Range-Wise Metric (New!)**
In [SSF paper](https://arxiv.org/abs/2501.17821), we introduce a new distance-based evaluation metric for scene flow estimation. Below is an example output for SSF with point_cloud_range to 204.8m and voxel_size=0.2m. Check more long-range result in [SSF paper](https://arxiv.org/abs/2501.17821).

| Distance  | Static    | Dynamic  | NumPointsStatic | NumPointsDynamic |
|-----------|----------|----------|-----------------|------------------|
| 0-35      | 0.00836  | 0.11546  | 3.33e+08        | 1.57e+07         |
| 35-50     | 0.00910  | 0.16805  | 4.40e+07        | 703125           |
| 50-75     | 0.01107  | 0.20448  | 3.25e+07        | 395398           |
| 75-100    | 0.01472  | 0.24133  | 1.31e+07        | 145281           |
| 100-inf   | 0.01970  | 0.30536  | 1.32e+07        | 171865           |
| **Mean**  | 0.01259  | 0.20693  | NaN             | NaN              |


### Submit result to public leaderboard
To submit your result to the public Leaderboard, if you select `data_mode=test`, it should be a zip file for you to submit to the leaderboard.
Note: The leaderboard result in DeFlow&SeFlow main paper is [version 1](https://eval.ai/web/challenges/challenge-page/2010/evaluation), as [version 2](https://eval.ai/web/challenges/challenge-page/2210/overview) is updated after DeFlow&SeFlow.

```bash
# since the env may conflict we set new on deflow, we directly create new one:
conda create -n py37 python=3.7
conda activate py37
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
# (feed-forward): load ckpt
python save.py checkpoint=/home/kin/seflow_best.ckpt dataset_path=/home/kin/data/av2/preprocess_v2/sensor/vis
# (optimization-based): change another model by passing model name.
python eval.py model=nsfp dataset_path=/home/kin/data/av2/h5py/demo/val

# The output of above command will be like:
Model: DeFlow, Checkpoint from: /home/kin/model_zoo/v2/seflow_best.ckpt
We already write the flow_est into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:
python tools/visualization.py --res_name 'seflow_best' --data_dir /home/kin/data/av2/preprocess_v2/sensor/vis
Enjoy! ^v^ ------ 

# Then run the command in the terminal:
python tools/visualization.py --res_name 'seflow_best' --data_dir /home/kin/data/av2/preprocess_v2/sensor/vis
```

https://github.com/user-attachments/assets/f031d1a2-2d2f-4947-a01f-834ed1c146e6

For exporting easy comparsion with ground truth and other methods, we also provided multi-visulization open3d window:
```bash
python tools/visualization.py --mode mul --res_name "['flow', 'seflow_best']" --data_dir /home/kin/data/av2/preprocess_v2/sensor/vis
```

Or another way to interact with [rerun](https://github.com/rerun-io/rerun) but please only vis scene by scene, not all at once.

```bash
python tools/visualization_rerun.py --data_dir /home/kin/data/av2/h5py/demo/train --res_name "['flow', 'deflow']"
```

https://github.com/user-attachments/assets/07e8d430-a867-42b7-900a-11755949de21


## Cite Us

[*OpenSceneFlow*](https://github.com/KTH-RPL/OpenSceneFlow) is originally designed by [Qingwen Zhang](https://kin-zhang.github.io/) from DeFlow and SeFlow. 
It is actively maintained and developed by the community (ref. below works).
If you find it useful, please cite our works:

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
@article{zhang2025himo,
  title={{HiMo}: High-Speed Objects Motion Compensation in Point Cloud},
  author={Zhang, Qingwen and Khoche, Ajinkya and Yang, Yi and Ling, Li and Mansouri, Sina Sharif and Andersson, Olov and Jensfelt, Patric},
  journal={IEEE Transactions on Robotics}, 
  year={2025},
  volume={41},
  pages={5896-5911},
  doi={10.1109/TRO.2025.3619042}
}
@inproceedings{zhang2025deltaflow,
  title={{DeltaFlow}: An Efficient Multi-frame Scene Flow Estimation Method},
  author={Zhang, Qingwen and Zhu, Xiaomeng and Zhang, Yushan and Cai, Yixi and Andersson, Olov and Jensfelt, Patric},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=T9qNDtvAJX}
}
@misc{zhang2025teflow,
  title={{TeFlow}: Enabling Multi-frame Supervision for Feed-forward Scene Flow Estimation},
  author={Zhang, Qingwen and Jiang, Chenhan and Zhu, Xiaomeng and Miao, Yunqi and Zhang, Yushan and Andersson, Olov and Jensfelt, Patric},
  year={2025},
  url={https://openreview.net/forum?id=h70FLgnIAw}
}
```

And our excellent collaborators works contributed to this codebase also:

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
@inproceedings{khoche2025ssf,
  title={{SSF}: Sparse Long-Range Scene Flow for Autonomous Driving},
  author={Khoche, Ajinkya and Zhang, Qingwen and Sanchez, Laura Pereira and Asefaw, Aron and Mansouri, Sina Sharif and Jensfelt, Patric},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)}, 
  year={2025},
  pages={6394-6400},
  doi={10.1109/ICRA55743.2025.11128770}
}
@inproceedings{lin2025voteflow,
  title={VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow},
  author={Lin, Yancong and Wang, Shiming and Nan, Liangliang and Kooij, Julian and Caesar, Holger},
  booktitle={CVPR},
  year={2025},
}
```

Thank you for your support! ❤️
Feel free to contribute your method and add your bibtex here by pull request!

❤️: [BucketedSceneFlowEval](https://github.com/kylevedder/BucketedSceneFlowEval); [Pointcept](https://github.com/Pointcept/Pointcept); [OpenPCSeg](https://github.com/BAI-Yeqi/OpenPCSeg); [ZeroFlow](https://github.com/kylevedder/zeroflow) ...
