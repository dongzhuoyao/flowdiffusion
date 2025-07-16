
#https://github.com/NVlabs/Sana/issues/215#issuecomment-2749985573



# Sampling

python sample_acc_kl.py model=cdit_b2_learnsigma data=cub200_256_cond model.params.in_channels=4 use_latent=1 dynamic=fmshortcut data.batch_size=64 debug=0 ckpt=./outputs/dyndit_cdit_b2_learnsigma_cub200_256_cond_bs64/2024-10-07_10-32-40_None/checkpoints/0030000.pt


# Train


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  --mixed_precision bf16  --num_processes 4 --num_machines 1 --multi_gpu --main_process_ip 127.0.0.1 --main_process_port 8868 train_acc_kl.py model=cdit_b2 data=cub200_256_uncond model.params.in_channels=4 use_latent=1 dynamic=fm  data.batch_size=64 optim.lr=1e-4  debug=0



CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch  --mixed_precision bf16  --num_processes 4 --num_machines 1 --multi_gpu --main_process_ip 127.0.0.1 --main_process_port 8868 train_acc_kl.py model=cdit_b2_shortcut data=cub200_256_uncond model.params.in_channels=4 use_latent=1 dynamic=fmshortcut  data.batch_size=64 optim.lr=1e-4  debug=0






CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3 accelerate launch  --mixed_precision bf16  --num_processes  2 --num_machines 1 --multi_gpu --main_process_ip 127.0.0.1 --main_process_port 8868 train_acc_kl.py mixed_precision=bf16 model=cdit_b2 model.params.use_shortcut=False data=cub200_256_cond model.params.in_channels=4 use_latent=0  dynamic=fm  data.batch_size=64 optim.lr=1e-4  debug=0


CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3 accelerate launch  --mixed_precision bf16  --num_processes  2 --num_machines 1 --multi_gpu --main_process_ip 127.0.0.1 --main_process_port 8868 train_acc_kl.py mixed_precision=bf16 model=cdit_b2 model.params.use_shortcut=True data=cub200_256_cond model.params.in_channels=4 use_latent=0 dynamic=fmshortcut  data.batch_size=64 optim.lr=1e-4



# Env

```
conda create --name hsi  --clone discretediffusion
```

## Train
python train_acc_kl.py model=cdit_s2 data=cub200_256_cond model.params.in_channels=4 use_latent=1 dynamic=sit debug=0

## Evaluation (Generation)

python sample_acc_kl.py model=cdit_b2_learnsigma data=cub200_256_cond model.params.in_channels=4 use_latent=1 dynamic=dyndit data.batch_size=64 optim.lr=4e-4 debug=0 ckpt=./outputs/dyndit_cdit_b2_learnsigma_cub200_256_cond_bs64/2024-10-07_10-32-40_None/checkpoints/0030000.pt


# For backbone design, we have several choices built on transformers. 

- 1). [muse-maskgit-pytorch](https://github.com/lucidrains/muse-maskgit-pytorch/blob/6df7f33bcd33ba28a2f682d5bd293e4f8a513e6c/muse_maskgit_pytorch/muse_maskgit_pytorch.py#L199)

- 2). ImageBert following titok.

- 3). UVITBert following titok



# Dataset Preparation 


## Pokemon dataset
kaggle datasets download lantian773030/pokemonclassification

unzip pokemonclassification.zip

## Environment setup

```
conda create -n isc python=3.11
conda activate isc
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install  torchdiffeq  matplotlib h5py timm diffusers accelerate loguru blobfile ml_collections wandb absl-py
pip install hydra-core opencv-python torch-fidelity webdataset einops pytorch_lightning
pip install torchmetrics --upgrade
pip install moviepy imageio 
pip install  scikit-learn --upgrade 
pip install diffusers  open_clip-torch einops omegaconf webdataset
```

# 
real_subsampling_factor=8 for Taichi, =3 for other datasets, https://github.com/Vchitect/Latte/issues/115


#imageio.save videos following:

https://github.com/Vchitect/Latte/blob/2fccaf686e13ca5214efd2ca20a14bd894d62f92/sample/sample_ddp.py#L176


# 
you need first save as video, then extra frames from the video, then calculate the FVD based on the frames. see https://github.com/Vchitect/Latte/issues/117

