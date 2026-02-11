# Unleashing the Power of Graph Data Augmentation on Covariate Distribution Shift
We provide a detailed code for "Unleashing the Power of Graph Data Augmentation on Covariate Distribution Shift".

Yongduo Sui, Qitian Wu, Jiancan Wu, Qing Cui, Longfei Li, Jun Zhou, Xiang Wang, Xiangnan He.

In NeurIPS 2023: https://openreview.net/forum?id=hIGZujtOQv.

## Installations
Main packages: PyTorch, Pytorch Geometric, OGB.
```bash
# Docker
docker run -it --gpus all --name aia -v /home/jylim/project:/workspace --security-opt seccomp=unconfined pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# Apt-get
printf 'APT::Sandbox::User "root";\nAPT::Sandbox::Seccomp "false";\n' > /etc/apt/apt.conf.d/99no-sandbox

# Conda
conda init bash
exec bash

apt-get update && apt-get install -y git

git clone https://github.com/limlimlim00/AIA.git
cd AIA

conda create -n aia310 python=3.10
conda activate aia310

pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# PyG core + CUDA extension
pip install torch-geometric
pip install "scipy<2"
pip install --no-index \
  --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html \
  pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv

pip install ogb munch ruamel.yaml typed-argument-parser cilog tensorboard gdown rdkit matplotlib
```

## Preparations
Please download the graph OOD datasets and OGB datasets as described in the original paper. 
Create a folder ```dataset```, and then put the datasets into ```dataset```. Then modify the path by specifying ```--data_dir your/path/dataset```.


## Commands
 We use the NVIDIA GeForce RTX 3090 (24GB GPU) to conduct all our experiments.
 To run the code on CMNIST, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_syn_it.py \
--trails 10 \
--dataset cmnist \
--emb_dim 300 \
--epochs 100 \
--cau_gamma 0.5 \
--adv_gamma_node 1.0 \
--adv_gamma_edge 0.8 \
--adv_dis 0.2 \
--adv_reg 0.5 \
--cau_reg 1.0 \
--causaler_lr 0.001 \
--attacker_lr 0.005 \
--test_epoch 10 --data_dir $DATA_DIR

```
 

 To run the code on Molbbbp, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_mol_it.py \
--trails 10 \
--domain scaffold \
--dataset ogbg-molbbbp \
--epochs 100 \
--emb_dim 64 \
--cau_gamma 0.5 \
--adv_dis 0.5 \
--adv_reg 0.5 \
--cau_reg 0.5 \
--causaler_lr 0.001 \
--attacker_lr 0.001 --data_dir $DATA_DIR
```

To run the code on Motif, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_syn_it.py \
--trails 10 \
--domain basis \
--dataset motif \
--epochs 100 \
--cau_gamma 0.5 \
--adv_gamma 1.0 \
--adv_gamma_edge 0.8 \
--adv_dis 0.2 \
--adv_reg 0.5 \
--cau_reg 1.0 \
--causaler_lr 0.001 \
--attacker_lr 0.005 \
--data_dir $DATA_DIR
```

 To run the code on Molhiv, please use the following command:
 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_adv_mol_it.py \
--trails 10 \
--domain size \
--dataset hiv \
--epochs 100 \
--emb_dim 128 \
--cau_gamma 0.1 \
--adv_gamma_node 1.0 \
--adv_gamma_edge 1.0 \
--adv_dis 1.5 \
--adv_reg 0.5 \
--cau_reg 0.5 \
--causaler_lr 0.01 \
--attacker_lr 0.01 \
--data_dir $DATA_DIR
```

## Citation
If you use our codes or checkpoints, please cite our paper:
```
@inproceedings{sui2023unleashing,
    title={Unleashing the Power of Graph Data Augmentation on Covariate Distribution Shift},
    author={Sui, Yongduo and Wu, Qitian and Wu, Jiancan and Cui, Qing and Li, Longfei and Zhou, Jun and Wang, Xiang and He, Xiangnan},
    booktitle={NeurIPS},
    year={2023},
    url={https://openreview.net/pdf?id=hIGZujtOQv}
}
```



