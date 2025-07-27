


# What you need to add when you want to train on your own task


1. add your own config in config/data/YOUR_DATASET_NAME.yaml
2. add your own model in models/YOUR_MODEL_NAME.py, we have prepared some models for you, you can use them as a reference. 
3. add your own dataloader in datasets_wds/YOUR_DATASET_NAME.py
4. add your new dynamic in dynamics/YOUR_DYNAMIC_NAME.py, we have prepared some dynamics for you, you can use them as a reference. we includes flow matching, diffusion, shortcut model, etc.
5. make some small changes in training script in train_acc_kl.py, you can use the training script as a reference.




# Environment Preparation


```
conda create -n fd python=3.11
conda activate fd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install  torchdiffeq  matplotlib h5py timm diffusers accelerate loguru blobfile ml_collections wandb absl-py wids
pip install hydra-core opencv-python torch-fidelity webdataset einops pytorch_lightning
pip install torchmetrics --upgrade
pip install moviepy imageio 
pip install  scikit-learn --upgrade 
pip install diffusers  open_clip-torch einops omegaconf webdataset
```



# multi-gpu Train

```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3 accelerate launch  --mixed_precision bf16  --num_processes  2 --num_machines 1 --multi_gpu --main_process_ip 127.0.0.1 --main_process_port 8868 train_acc_kl.py mixed_precision=bf16 model=cdit_b2 model.params.use_shortcut=False data=cub200_256_cond model.params.in_channels=4 use_latent=0  dynamic=fm  data.batch_size=64 optim.lr=1e-4  debug=0
```

# Single-gpu Train on CUB about Flow Matching

```
CUDA_VISIBLE_DEVICES=2 accelerate launch  --mixed_precision bf16  --num_processes  1 --num_machines 1 --main_process_ip 127.0.0.1 --main_process_port 8868 train_acc_kl.py mixed_precision=bf16 model=cdit_s2 model.params.use_shortcut=False data=cub200_256_cond_toy model.params.in_channels=4 use_latent=0  dynamic=fm  data.batch_size=64 optim.lr=1e-4  debug=0
```

# Single-gpu Train on CUB about Flow Matching with Shortcut

```
CUDA_VISIBLE_DEVICES=2 accelerate launch  --mixed_precision bf16  --num_processes  1 --num_machines 1 --main_process_ip 127.0.0.1 --main_process_port 8868 train_acc_kl.py mixed_precision=bf16 model=cdit_s2_sc model.params.use_shortcut=True data=cub200_256_cond_toy model.params.in_channels=4 use_latent=0  dynamic=fm  data.batch_size=64 optim.lr=1e-4  debug=0
```



# Sampling

```
python sample_acc_kl.py model=cdit_b2_learnsigma data=cub200_256_cond model.params.in_channels=4 use_latent=1 dynamic=fmshortcut data.batch_size=64 debug=0 ckpt=./outputs/dyndit_cdit_b2_learnsigma_cub200_256_cond_bs64/2024-10-07_10-32-40_None/checkpoints/0030000.pt
```




# Dataset Preparation 

## CUB dataset
download CUB dataset from [https://www.vision.caltech.edu/datasets/cub_200_2011/](https://www.vision.caltech.edu/datasets/cub_200_2011/) and put CUB_200_2011.tgz in data/CUB_200_2011
run the following script to make the webdataset dataset
```
python datasets_wds/make_wds_cub256_raw_varysize.py 
```




## Pokemon dataset
kaggle datasets download lantian773030/pokemonclassification
unzip pokemonclassification.zip





