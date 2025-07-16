import torch
import torchvision.transforms as transforms
import wids

import wandb
from wandb_utils import array2grid_pixel

cifar10_classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
cityscapes_big8 = [
    "Flat",
    "Human",
    "Vehicle",
    "Construction",
    "Object",
    "Nature",
    "Sky",
    "Void",
]

cub_classes = ["cub_" + str(i) for i in range(200)]


import torch.distributed as dist

import logging


def print_rank_0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
            # logging.info(*args, **kwargs)
    else:
        print(*args, **kwargs)


def wandb_visual_dict(wandb_key, visual_tensor, is_video, num=16, captions=None):
    if captions is None:
        captions = ["null caption" for _ in range(num)]
    if is_video:
        b, t, c, w, h = visual_tensor.shape
        visual_tensor = visual_tensor.cpu().numpy()
        return {
            wandb_key: wandb.Video(visual_tensor[:num]),
        }
    else:
        b, c, w, h = visual_tensor.shape
        return {
            wandb_key: wandb.Image(array2grid_pixel(visual_tensor[:num])),
        }


def get_version_number():
    # return "v0"
    # return "v1"  # fix a bug in delta_max annealing schedule, sometimes, dt can be 0.5 in progress=0.2, which is abnormal
    #return "v2"  # fix a bug in sampling , I forget multply dt 
    #return "v2.1"  # fix showing flow-loss,bst-loss
    # make the t,dt of flowloss totally continuous 
    return "v3"  # fix showing flow-loss,bst-loss
def has_label(dataset_name):
    if dataset_name.startswith("ffs"):
        return False
    else:
        return True


def get_dataset_id2label(dataset_name):
    if "imagenet" in dataset_name:
        imagenet_id2realname = open("./datasets_wds/imagenet1k_name.txt").readlines()
        imagenet_id2realname = [
            _cls.strip().split()[-1] for _cls in imagenet_id2realname
        ]
        return imagenet_id2realname
    elif "cifar10" in dataset_name:
        return cifar10_classes
    elif "cs" in dataset_name:
        return cityscapes_big8
    elif "cub" in dataset_name:
        return cub_classes
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_wid_dl(
    shuffle=True,
    num_workers=1,
    batch_size=4,
    json_path="./data/imagenet256_raw_wds_train.json",
):
    wids_dataset = wids.ShardListDataset(json_path)  # keep=True)

    class _WIDSDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.transform_train = transforms.Compose(
                [
                    # transforms.ToTensor(),
                    transforms.PILToTensor(),
                ]
            )

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]
            _img = sample[".image.jpg"]
            _img = self.transform_train(_img)
            _cls_id = int(sample[".cls_id.cls"])
            return _img, _cls_id  # , _cls_name

    dl = torch.utils.data.DataLoader(
        _WIDSDataset(wids_dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dl


def get_inf_wid_dl_imglabel(
    args,
    batch_size,
    shuffle=True,
    num_workers=4,
    device=None,
):
    if "imagenet" in args.data.name:
        _dl = get_wid_dl(
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            json_path=args.data.tar_base_wid_json,
        )
    elif "cifar10" in args.data.name:
        _dl = get_wid_dl(
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            json_path=args.data.tar_base_wid_json,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.data.name}")

    def inifite_generator():
        while True:
            for _img, _label in _dl:
                # image range [0, 255], label start from 0
                yield _img.to(device), _label.to(device)

    return inifite_generator()


def get_inf_wid_dl_imgonly(args, batch_size, device, shuffle=True, num_workers=4):
    gen = get_inf_wid_dl_imglabel(args, batch_size, shuffle, num_workers, device=device)
    for img, cls_id in gen:
        yield img.to(device)


if __name__ == "__main__":
    pass
