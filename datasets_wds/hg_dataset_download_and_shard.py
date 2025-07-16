from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import webdataset as wds
import os
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from diffusers import AutoencoderKL
from tqdm import tqdm
import numpy as np
import fire
import torch.nn.functional as F
import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1800"
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils_repa import preprocess_raw_image

class FFHQDataset(Dataset):
    """
    Dataset class for loading ImageNet-1k from Hugging Face
    """

    def __init__(self,  cache_dir="./data", transform=None):
        """
        Initialize the dataset.

        Args:
            split (str): Dataset split to use ('train' or 'validation')
            transform (callable, optional): Optional transform to apply to the images
        """
        self.dataset = load_dataset("marcosv/ffhq-dataset", split="train", cache_dir=cache_dir)
        self.transform = transform

      

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to get

        Returns:
            tuple: (image, label) where label is the class index
        """
        item = self.dataset[idx]
        image = item["image"]
    
        if self.transform:
            image = self.transform(image)

        return image
    
class ImageNetDataset(Dataset):
    """
    Dataset class for loading ImageNet-1k from Hugging Face
    """

    def __init__(self, split="train", cache_dir="./data", transform=None):
        """
        Initialize the dataset.

        Args:
            split (str): Dataset split to use ('train' or 'validation')
            transform (callable, optional): Optional transform to apply to the images
        """
        self.dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split=split, cache_dir=cache_dir)
        self.transform = transform

        # Map class labels to indices (0-999)
        self.classes = sorted(list(set(self.dataset["label"])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to get

        Returns:
            tuple: (image, label) where label is the class index
        """
        item = self.dataset[idx]
        image = item["image"]
        label = self.class_to_idx[item["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_imagenet_dataloader(batch_size: int, num_workers: int = 8):
    """
    Create DataLoaders for the ImageNet-1k dataset.

    Args:
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of worker processes for data loading

    Returns:
        tuple: (train_loader, val_loader)
    """

    # All images from the dataset are 256x256 resolution
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageNetDataset(split="train", transform=transform)

    # Create dataloaders
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader



def get_ffhq_dataloader(batch_size: int, num_workers: int = 8,resize_to: int = 256):
    transform = transforms.Compose([transforms.Resize(resize_to), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = FFHQDataset(transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader


use_tc = True  # torch.compile

def kl_get_vae(device="cuda"):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    if use_tc:
        vae = torch.compile(vae,mode="max-autotune", fullgraph=True)

    vae.eval()
    vae = vae.to(device)
    
    vae_encode = vae.encode
    return vae_encode

@torch.no_grad()
def encode_latents(vae_encode, images, mini_bs=25):
    # images: torch.Tensor, shape (B, 3, 256, 256), range [-1, 1]
    assert images.min() >= -1 and images.max() <= 1
    latents = []
    for i in range(0, len(images), mini_bs):
        _img = images[i:i+mini_bs]
        _lat = vae_encode(_img).latent_dist.sample().mul_(0.18215)
        latents.append(_lat)
    return torch.cat(latents, dim=0)

@torch.no_grad()
def encode_dino(encoder_ssl, images,encoder_type="dinov2"):
    # images: torch.Tensor, shape (B, 3, 256, 256), range [-1, 1]
    assert images.min() >= -1 and images.max() <= 1
    image = (images * 0.5 + 0.5).clamp(0, 1)*255.0
    raw_image_ = preprocess_raw_image(image, encoder_type)
    if use_tc:
        _forward_features = torch.compile(encoder_ssl.forward_features,mode="max-autotune", fullgraph=True)
    else:
        _forward_features = encoder_ssl.forward_features
    z =_forward_features(raw_image_)
    if "mocov3" in encoder_type:
        z = z = z[:, 1:]
    if "dinov2" in encoder_type:
        z = z["x_norm_patchtokens"]
    z = z.cpu().numpy()
    return z 

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def shard_imagenet(batch_size: int, num_workers: int = 8, include_lowz: bool = False, downsample_ratio: int = 4, save_image: bool = False,save_dino: bool = False,encoder_type="dinov2-vit-b", max_tar_size=0.1, output_dir="./data/imagenet_1k_256x256_latents"):
    output_dir =output_dir+"_ds" +str(downsample_ratio) +f"_img{int(save_image)}dino{int(save_dino)}"
    dataset = ImageNetDataset(split="train", transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    rank, world_size = setup_distributed()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(rank)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler,
    )

    vae_encode = kl_get_vae(device=device)
    
    #vae = torch.nn.parallel.DistributedDataParallel(vae, device_ids=[rank])
    os.makedirs(output_dir, exist_ok=True)

    if rank == 0:
        writer = wds.ShardWriter(
            os.path.join(output_dir, "shard_%06d.tar"),
            maxcount=1e6,
            maxsize=1e9 * max_tar_size,
        )
        sample_idx = 0

    if save_dino:
        from utils_repa import load_encoders
        encoders, encoder_types, architectures = load_encoders(enc_type=encoder_type,device=device)
        encoder_ssl = encoders[0]
        encoder_ssl = encoder_ssl.to(device)
    try:
        for images, labels in tqdm(dataloader) if rank == 0 else dataloader:
            images = images.to(device)
            latents = encode_latents(vae_encode, images)
            if save_dino:
                dino_features = encode_dino(encoder_ssl, images,encoder_type=encoder_type)
            else:
                dino_features = np.zeros_like(latents.cpu().numpy())
            latents_np = latents.cpu().numpy()
            labels_np = labels.cpu().numpy()
            images_np = images.cpu().numpy()
            if include_lowz:
                images_low_res = F.interpolate(images, scale_factor=1/downsample_ratio, mode="bilinear", align_corners=False)
                images_low_res = F.interpolate(images_low_res, scale_factor=downsample_ratio, mode="bilinear", align_corners=False)
                latents_low_res = encode_latents(vae_encode, images_low_res)
                latents_low_res_np = latents_low_res.cpu().numpy()
            else:
                latents_low_res_np = np.zeros_like(latents_np)#fake low res latents

            # Gather all latents, labels, and images to rank 0
            gathered_latents = [None for _ in range(world_size)]
            gathered_labels = [None for _ in range(world_size)]
            gathered_images = [None for _ in range(world_size)]
            gathered_latents_low_res = [None for _ in range(world_size)]
            gathered_dino_features = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_dino_features, dino_features)
            dist.all_gather_object(gathered_latents, latents_np)
            dist.all_gather_object(gathered_labels, labels_np)
            dist.all_gather_object(gathered_images, images_np)
            dist.all_gather_object(gathered_latents_low_res, latents_low_res_np)

            if rank == 0:
                for gpu_latents, gpu_latents_low_res, gpu_labels, gpu_images, gpu_dino_features in zip(gathered_latents, gathered_latents_low_res, gathered_labels, gathered_images, gathered_dino_features):
                    for latent, latent_low_res, label, image, dino_feature in zip(gpu_latents, gpu_latents_low_res, gpu_labels, gpu_images, gpu_dino_features):
                        sample = {
                            "__key__": f"{sample_idx:08d}",
                            "latent.npy": latent.astype("float32"),
                            "cls_id.cls": int(label),
                        }
                        if include_lowz:
                            assert latent_low_res.shape == latent.shape
                            sample["latent_lowz.npy"] = latent_low_res.astype("float32")
                        if save_image:
                            # image shape should be (3, 256, 256)
                            if isinstance(image, np.ndarray):
                                if image.shape[0] == 3:  # (3, H, W)
                                    image_tensor = torch.from_numpy(image)
                                elif image.shape[-1] == 3:  # (H, W, 3)
                                    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
                                else:
                                    raise ValueError(f"Unexpected image shape: {image.shape}")
                            else:
                                image_tensor = image  # already a torch tensor

                            # Unnormalize from [-1, 1] to [0, 1]
                            image_tensor = (image_tensor * 0.5 + 0.5).clamp(0, 1)
                            image_pil = transforms.ToPILImage()(image_tensor)
                            import io
                            buf = io.BytesIO()
                            image_pil.save(buf, format="JPEG", quality=95)
                            image_bytes = buf.getvalue()
                            sample["image.jpg"] = image_bytes
                        if save_dino:
                            sample["dino_feature.npy"] = dino_feature.astype("float32")
                        
                            
                        
                        writer.write(sample)
                        sample_idx += 1
    finally:
        if rank == 0:
            writer.close()
        cleanup_distributed()



def shard_ffhq(batch_size: int, num_workers: int = 8,save_image: bool = False, max_tar_size=0.02, include_lowz=False, downsample_ratio=4, original_size=256, output_dir="./data/ffhq_256x256_latents"):
    dataset = FFHQDataset(transform=transforms.Compose([
        transforms.Resize(original_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))
    
    rank, world_size = setup_distributed()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(rank)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=sampler,
    )

    vae = kl_get_vae(device=device)
    os.makedirs(output_dir, exist_ok=True)

    if rank == 0:
        writer = wds.ShardWriter(
            os.path.join(output_dir, "shard_%06d.tar"),
            maxcount=1e6,
                maxsize=1e9 * max_tar_size,
        )
        sample_idx = 0

    try:
        for images in tqdm(dataloader) if rank == 0 else dataloader:
            images = images.to(device)
            latents = encode_latents(vae_encode, images)
            latents_np = latents.cpu().numpy()
            if include_lowz:
                images_low_res = F.interpolate(images, scale_factor=1/downsample_ratio, mode="bilinear", align_corners=False)
                images_low_res = F.interpolate(images_low_res, scale_factor=downsample_ratio, mode="bilinear", align_corners=False)
                latents_low_res = encode_latents(vae_encode, images_low_res)
                latents_low_res_np = latents_low_res.cpu().numpy()
            else:
                latents_low_res_np = np.zeros_like(latents_np)#fake low res latents
            images_np = images.cpu().numpy()

            # Gather all latents, labels, and images to rank 0
            gathered_latents = [None for _ in range(world_size)]
            
            gathered_images = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_latents, latents_np)
            dist.all_gather_object(gathered_images, images_np)
            if include_lowz:
                gathered_latents_low_res = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_latents_low_res, latents_low_res_np)

            if rank == 0:
                for gpu_latents, gpu_latents_low_res, gpu_images in zip(gathered_latents,gathered_latents_low_res,  gathered_images):
                    for latent, latent_low_res, image in zip(gpu_latents, gpu_latents_low_res, gpu_images):
                        sample = {
                            "__key__": f"{sample_idx:08d}",
                            "latent.npy": latent.astype("float32"),
                        }
                        if include_lowz:
                            assert latent_low_res.shape == latent.shape
                            sample["latent_lowz.npy"] = latent_low_res.astype("float32")
                        if save_image:
                            # image shape should be (3, 256, 256)
                            if isinstance(image, np.ndarray):
                                if image.shape[0] == 3:  # (3, H, W)
                                    image_tensor = torch.from_numpy(image)
                                elif image.shape[-1] == 3:  # (H, W, 3)
                                    image_tensor = torch.from_numpy(image).permute(2, 0, 1)
                                else:
                                    raise ValueError(f"Unexpected image shape: {image.shape}")
                            else:
                                image_tensor = image  # already a torch tensor

                            # Unnormalize from [-1, 1] to [0, 1]
                            image_tensor = (image_tensor * 0.5 + 0.5).clamp(0, 1)
                            image_pil = transforms.ToPILImage()(image_tensor)
                            import io
                            buf = io.BytesIO()
                            image_pil.save(buf, format="JPEG", quality=95)
                            image_bytes = buf.getvalue()
                            sample["image.jpg"] = image_bytes

                        
                        writer.write(sample)
                        sample_idx += 1
    finally:
        if rank == 0:
            writer.close()
        cleanup_distributed()


class Runner:
    def __init__(self):
        pass

    def imagenet(self, save_image: bool = False, save_dino: bool = False,max_tar_size: float = 0.1):
        """Shard the ImageNet-1k dataset"""
        shard_imagenet(batch_size=64, num_workers=8, output_dir="./data/imagenet_1k_256x256_latents", save_image=save_image, save_dino=save_dino,max_tar_size=max_tar_size)
    
    def imagenet_cfm(self, save_image: bool = False, save_dino: bool = False,downsample_ratio: int = 4,max_tar_size: float = 0.1):
        """Shard the ImageNet-1k dataset"""
        shard_imagenet(batch_size=64, num_workers=8, include_lowz=True, downsample_ratio=downsample_ratio, save_image=save_image, save_dino=save_dino, max_tar_size=max_tar_size, output_dir="./data/imagenet_1k_256x256_latents_cfm")

    def ffhq(self):
        """Shard the FFHQ dataset"""
        shard_ffhq(batch_size=64, num_workers=8, output_dir="./data/ffhq_256x256_latents")

    def ffhq_cfm(self):
        """Shard the FFHQ dataset"""
        shard_ffhq(batch_size=64, num_workers=8, include_lowz=True,downsample_ratio=4, original_size=256, output_dir="./data/ffhq_256x256_latents_cfm")

if __name__ == "__main__":
    fire.Fire(Runner)
    #torchrun --nproc_per_node=4 datasets_wds/hg_dataset_download_and_shard.py imagenet --save_dino=1 --save_image=0 --max_tar_size=1
    # torchrun --nproc_per_node=4 datasets_wds/hg_dataset_download_and_shard.py imagenet
    #torchrun --nproc_per_node=4  datasets_wds/hg_dataset_download_and_shard.py imagenet_cfm
    #torchrun --nproc_per_node=4  datasets_wds/hg_dataset_download_and_shard.py ffhq
    #torchrun --nproc_per_node=4  datasets_wds/hg_dataset_download_and_shard.py ffhq_cfm 
    # torchrun --nproc_per_node=4  datasets_wds/hg_dataset_download_and_shard.py imagenet_cfm --downsample_ratio=4 --save_dino=1 --save_image=0 --max_tar_size=1 
    # torchrun --nproc_per_node=4  datasets_wds/hg_dataset_download_and_shard.py imagenet_cfm --downsample_ratio=8 --save_dino=1 --save_image=0 --max_tar_size=1 
    