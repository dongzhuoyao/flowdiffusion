import os
import webdataset as wds
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
import io
import numpy as np
from tqdm import tqdm
import numpy as np
from diffusers import AutoencoderKL
import hydra
import torch.distributed as dist

# Source and destination directories



# Constants
TARGET_SIZE = 256
MAX_TAR_SIZE = 50 * 1024 * 1024  # 50MB per tar file
NUM_WORKERS = 1
BS = 64



def kl_get_vae(device="cuda"):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    # stabilityai/sd-vae-ft-ema
    vae.eval()
    return vae



@torch.no_grad()
def encode_fn(vae, img, mini_bs=25):
    #img = img / 255.0
    #img = img * 2.0 - 1.0
    assert img.min() >= -1 and img.max() <= 1

    ############################################################
    for i in range(0, len(img), mini_bs):
        _img = img[i : i + mini_bs]
        assert _img.min() >= -1 and _img.max() <= 1
        _indices = vae.encode(_img).latent_dist.sample().mul_(0.18215)
        if i == 0:
            indices = _indices
        else:
            indices = torch.cat([indices, _indices], dim=0)
    ############################################################
    return indices


vae = kl_get_vae()

def encoder_batch_images(images):
    return encode_fn(vae, images)


vis_once_flag=True

def process_sample(sample, vae, device="cuda"):
    """Process a batch of samples: resize image.png, compute VAE latents, and keep all other keys unchanged."""
    new_samples = []

    # sample['image.png'] is a list of bytes (batch)
    images = [Image.open(io.BytesIO(img_bytes)).convert("RGB") for img_bytes in sample['image.png']]
    
    # Resize and preprocess images
    transform = T.Compose([
        T.Resize(TARGET_SIZE, antialias=True),
        T.CenterCrop(TARGET_SIZE),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # Scale to [-1, 1] as expected by VAE
    ])
    image_tensors = torch.stack([transform(img) for img in images]).to(device)  # Shape: (B, 3, H, W)
    global vis_once_flag
    

    # Compute VAE latents
    with torch.no_grad():
        latents = encoder_batch_images(image_tensors)    
    latents_np = latents.cpu().numpy()

    

    # For each sample in the batch, create a new_sample dict
    batch_size = len(images)
    for i in range(batch_size):
        new_sample = {}
        # Save image back to bytes
        image_pil = T.ToPILImage()(image_tensors[i].cpu().clamp(-1, 1) * 0.5 + 0.5)
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format='PNG')
        new_sample['image.png'] = image_bytes.getvalue()
        new_sample['latent.npy'] = latents_np[i].astype(np.float32)
        

        # Copy all other keys, converting tensors to numpy arrays if needed
        for key, value in sample.items():
            if key != 'image.png':
                if isinstance(value, torch.Tensor):
                    new_sample[key] = value[i].cpu().numpy()
                elif isinstance(value, list):
                    new_sample[key] = value[i]
                else:
                    new_sample[key] = value

        if vis_once_flag:
            print("vis_once_flag", vis_once_flag)
            for i in range(new_sample['latent.npy'].shape[0]):
                print("latents.max()", new_sample['latent.npy'][i].max(), "latents.min()", new_sample['latent.npy'][i].min(), "std", new_sample['latent.npy'][i].std(), "latents.shape", new_sample['latent.npy'][i].shape, "latents.dtype", new_sample['latent.npy'][i].dtype)
            vis_once_flag=False
            img_vis =vae.decode(new_sample['latent.npy'][0].unsqueeze(0)).sample
            img_vis = img_vis.cpu().clamp(-1, 1) * 0.5 + 0.5
            img_vis = T.ToPILImage()(img_vis.squeeze(0))
            img_vis.save("img_vis.png")
            print("saved img_vis.png")
        new_samples.append(new_sample)

    return new_samples


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

@hydra.main(config_path="../config", config_name="default", version_base=None)
def reshard_dataset(args):
    rank, world_size = setup_distributed()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(rank)

    tar_files = args.data.train_shards_path
    output_dir = os.path.dirname(tar_files) + "_precomputed_latent"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Rank {rank}: Resharding dataset from {tar_files} to {output_dir}")

    dataset = (
        wds.WebDataset(tar_files)
        .decode()
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    loader = DataLoader(
        dataset,
        batch_size=BS,
        num_workers=NUM_WORKERS,
        sampler=sampler,
        prefetch_factor=2
    )

    vae = kl_get_vae(device=device)
    current_count = 0

    if rank == 0:
        writer = wds.ShardWriter(
            os.path.join(output_dir, "shard_%06d.tar"),
            maxcount=1e6,
            maxsize=1e9 * 0.02,
        )

    try:
        for sample in tqdm(loader) if rank == 0 else loader:
            processed_samples = process_sample(sample, vae, device=device)
            # Gather all processed_samples to rank 0
            gathered_samples = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_samples, processed_samples)
            if rank == 0:
                # Flatten the list of lists
                for gpu_samples in gathered_samples:
                    for processed_sample in gpu_samples:
                        writer.write(processed_sample)
                        current_count += 1
    finally:
        if rank == 0:
            writer.close()
        cleanup_distributed()

    if rank == 0:
        print(f"Finished processing {current_count} samples")
        # Print statistics about created shards
        print("\nShard statistics:")
        for shard in sorted(os.listdir(output_dir)):
            if shard.endswith('.tar'):
                size_gb = os.path.getsize(os.path.join(output_dir, shard)) / (1024 * 1024 * 1024)
                print(f"{shard}: {size_gb:.2f} GB")

if __name__ == "__main__":
    reshard_dataset()
