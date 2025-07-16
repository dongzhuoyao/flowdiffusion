"""This file contains the definition of data loader using webdataset.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    https://github.com/huggingface/open-muse/blob/main/training/data.py
"""

import math
from typing import List, Union, Text
import webdataset as wds
from torch.utils.data import default_collate
from torchvision import transforms
import torch
import numpy as np
import io

try:
    from utils_common import print_rank_0
except ImportError:

    def print_rank_0(*args, **kwargs):
        print(*args, **kwargs)


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


class ImageTransform:
    def __init__(
        self,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        random_crop: bool = True,
        random_flip: bool = True,
        normalize_mean: List[float] = [0.0, 0.0, 0.0],
        normalize_std: List[float] = [1.0, 1.0, 1.0],
    ):
        """Initializes the WebDatasetReader with specified augmentation parameters.

        Args:
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.

        Raises:
            NotImplementedError: If the interpolation mode is not one of ["bicubic", "bilinear"].
        """
        train_transform = []
        interpolation = transforms.InterpolationMode.BICUBIC

        train_transform.append(
            transforms.Resize(
                resize_shorter_edge, interpolation=interpolation, antialias=True
            )  # https://github.com/openai/improved-diffusion/blob/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5/improved_diffusion/image_datasets.py#L87
        )
        if random_crop:
            train_transform.append(transforms.RandomCrop(crop_size))
        else:
            train_transform.append(transforms.CenterCrop(crop_size))
        if random_flip:
            train_transform.append(transforms.RandomHorizontalFlip())
        train_transform.append(transforms.ToTensor())
        # normalize_mean = [0, 0, 0] and normalize_std = [1, 1, 1] will normalize images into [0, 1],
        # normalize_mean = [0.5, 0.5, 0.5] and normalize_std = [0.5, 0.5, 0.5] will normalize images into [-1, 1].
        train_transform.append(transforms.Normalize(normalize_mean, normalize_std))

        self.train_transform = transforms.Compose(train_transform)
        self.eval_transform = transforms.Compose(
            [
                # Note that we always resize to crop_size during eval to ensure the results
                # can be compared against reference numbers on ImageNet etc.
                transforms.Resize(
                    crop_size, interpolation=interpolation, antialias=True
                ),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(normalize_mean, normalize_std),
            ]
        )
        print_rank_0(f"self.train_transform: {self.train_transform}")
        print_rank_0(f"self.eval_transform: {self.eval_transform}")


class SimpleImageDataset:
    def __init__(
        self,
        train_shards_path: Union[Text, List[Text]],
        eval_shards_path: Union[Text, List[Text]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_classes: int,
        hyper_feat_path: str = None,
        num_workers_per_gpu: int = 12,
        resize_shorter_edge: int = 256,
        crop_size: int = 256,
        use_latent: bool = False,
        random_crop=True,
        random_flip=True,
        normalize_mean: List[float] = [0.0, 0.0, 0.0],
        normalize_std: List[float] = [1.0, 1.0, 1.0],
        **kwargs,  #
    ):
        """Initializes the WebDatasetReader class.

        Args:
            train_shards_path: A string or list of string, path to the training data shards in webdataset format.
            eval_shards_path: A string or list of string, path to the evaluation data shards in webdataset format.
            num_train_examples: An integer, total number of training examples.
            per_gpu_batch_size: An integer, number of examples per GPU batch.
            global_batch_size: An integer, total number of examples in a batch across all GPUs.
            num_workers_per_gpu: An integer, number of workers per GPU.
            resize_shorter_edge: An integer, the shorter edge size to resize the input image to.
            crop_size: An integer, the size to crop the input image to.
            random_crop: A boolean, whether to use random crop augmentation during training.
            random_flip: A boolean, whether to use random flipping augmentation during training.
            normalize_mean: A list of float, the normalization mean used to normalize the image tensor.
            normalize_std: A list of float, the normalization std used to normalize the image tensor.
        """
        transform = ImageTransform(
            resize_shorter_edge=resize_shorter_edge,
            crop_size=crop_size,
            random_crop=random_crop,
            random_flip=random_flip,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

        train_processing_pipeline = build_train_processing_pipeline(
            transform, num_classes=num_classes, use_latent=use_latent
        )

        # Create train dataset and loader.
        pipeline = [
            wds.ResampledShards(train_shards_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(bufsize=5000, initial=1000),
            *train_processing_pipeline,
            wds.batched(
                per_gpu_batch_size, partial=False, collation_fn=default_collate
            ),
        ]

        # Each worker is iterating over the complete dataset.
        self._train_dataset = wds.DataPipeline(*pipeline)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=True,
            persistent_workers=False,  # I use yield to achieve infinite loop, no concept of epoch any more, set it to False
        )

    def train_dataset(self):
        return self._train_dataset

    def train_dataloader(self):
        return self._train_dataloader


def build_train_processing_pipeline(transform, num_classes=0, use_latent=False,save_image=False):
    # Build the set of keys to keep
    keys = set()
    if use_latent:
        keys.add("latent")
        keys.add("cls_id")

    if save_image:
        keys.add("image")
    print("keys", keys)

    # Build the rename mapping
    rename_map = { }

    if use_latent:#only save latent or image
        rename_map["latent"] = "latent.npy"
        rename_map["cls_id"] = "latent_lowz.npy"
    if save_image:
        rename_map["image"] = "jpg;png;jpeg;webp;image.jpg;image.png"
    print("rename_map", rename_map)

    # Build the pipeline
    pipeline = [
        wds.decode(
            wds.autodecode.ImageHandler(
                "pil", extensions=["webp", "png", "jpg", "jpeg"]
            ),   
        ),
        wds.rename(
            **rename_map,
            handler=wds.warn_and_continue,
        ),
       wds.map(filter_keys(keys)),
    ]

    if save_image:
        pipeline.append(
           wds.map_dict(
            image=transform.train_transform if hasattr(transform, "train_transform") else transform,
            handler=wds.warn_and_continue,
        ),
        wds.map_dict(
            image=lambda x: x * 2 - 1,  # [0,1] to [-1,1]
            handler=wds.warn_and_continue,
        ),
        )


    if use_latent:
        def decode_latent(arr):
            #arr = np.load(io.BytesIO(arr))
            #print("latent shape", arr.shape, "dtype", arr.dtype, "std", arr.std())
            return torch.from_numpy(arr)
        pipeline.append(
            wds.map_dict(
    latent=decode_latent,
    handler=wds.warn_and_continue,
)
        )

    return pipeline


if __name__ == "__main__":
    if True:
        dataloader = SimpleImageDataset(
            train_shards_path="./data/ffhq_256x256_latents_cfm/shard_{000000..000002}.tar",
            eval_shards_path="./data/ffhq_256x256_latents_cfm/shard_{000000..000002}.tar",
            num_train_examples=1000,
            per_gpu_batch_size=64,
            global_batch_size=64,
            num_workers_per_gpu=12,
            crop_size=256,
            num_classes=-1,
            random_crop=True,
            random_flip=True,
            use_latent=True,
        )

        for batch in dataloader.train_dataloader():
            print(batch.keys())
            #print(batch["image"].shape, batch["image"].max(), batch["image"].min())
            print( batch["latent"].shape, batch["latent"].max(), batch["latent"].min(),batch["latent"].mean(),batch["latent"].std())
            print( batch["cls_id"].shape, batch["cls_id"].max(), batch["cls_id"].min(),batch["cls_id"].mean(),batch["cls_id"].std())
            #print(batch["cls_id"].shape, batch["cls_id"].max(), batch["cls_id"].min())
            break
    elif False:
        dataloader = SimpleImageDataset(
            train_shards_path="./data/imagenet_1k_256x256_latents/shard_{000000..000002}.tar",
            eval_shards_path="./data/imagenet_1k_256x256_latents/shard_{000000..000002}.tar",
            num_train_examples=1000,
            per_gpu_batch_size=64,
            global_batch_size=64,
            num_workers_per_gpu=12,
            crop_size=256,
            num_classes=1001,
            random_crop=True,
            random_flip=True,
            use_latent=True,
            save_image=True,
        )
        for batch in dataloader.train_dataloader():
            print(batch.keys())
            #print(batch["image"].shape, batch["image"].max(), batch["image"].min())
            print( batch["latent"].shape, batch["latent"].max(), batch["latent"].min(),batch["latent"].mean(),batch["latent"].std())
            #print(batch["cls_id"].shape, batch["cls_id"].max(), batch["cls_id"].min())
            break
