


# Environment Preparation


```
conda create -n fd python=3.11
conda activate fd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install  torchdiffeq  matplotlib h5py timm diffusers accelerate loguru blobfile ml_collections wandb absl-py
pip install hydra-core opencv-python torch-fidelity webdataset einops pytorch_lightning
pip install torchmetrics --upgrade
pip install moviepy imageio 
pip install  scikit-learn --upgrade 
pip install diffusers  open_clip-torch einops omegaconf webdataset
```


# Train

```
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2,3 accelerate launch  --mixed_precision bf16  --num_processes  2 --num_machines 1 --multi_gpu --main_process_ip 127.0.0.1 --main_process_port 8868 train_acc_kl.py mixed_precision=bf16 model=cdit_b2 model.params.use_shortcut=False data=cub200_256_cond model.params.in_channels=4 use_latent=0  dynamic=fm  data.batch_size=64 optim.lr=1e-4  debug=0
```

# Sampling

```
python sample_acc_kl.py model=cdit_b2_learnsigma data=cub200_256_cond model.params.in_channels=4 use_latent=1 dynamic=fmshortcut data.batch_size=64 debug=0 ckpt=./outputs/dyndit_cdit_b2_learnsigma_cub200_256_cond_bs64/2024-10-07_10-32-40_None/checkpoints/0030000.pt
```




# Dataset Preparation 


## Pokemon dataset
kaggle datasets download lantian773030/pokemonclassification

unzip pokemonclassification.zip





