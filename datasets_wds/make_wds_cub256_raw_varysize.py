import sys

import PIL

sys.path.append("..")
import os
import argparse
import numpy as np
import webdataset as wds
from tqdm import tqdm
import torch
from PIL import Image
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
from torchvision.transforms import Resize, ToTensor
import torch
from torch.utils.data import DataLoader
from einops import rearrange
from omegaconf import OmegaConf


source_dir = f"./data/CUB_200_2011/images/"
wds_target_dir = f"./data/CUB_200_2011_wds_v2/"
wds_target_dir = os.path.expanduser(wds_target_dir)
global_feat_path = os.path.join(wds_target_dir, "global_feats.npy")

img_resize_dim = 256
latent_size = 32


class _Database(Dataset):
    def __init__(
        self,
    ):
        self.root = source_dir
        self.classes_names = []
        self.file_classes = []
        self.classes_names = os.listdir(self.root)
        for class_name in self.classes_names:
            file_names = os.listdir(os.path.join(self.root, class_name))
            for file_name in file_names:
                _cls = class_name.split(".")[0]
                _path = os.path.join(self.root, class_name, file_name)
                self.file_classes.append((_path, _cls))

        print("file_classes: ", len(self.file_classes))

    def __len__(self):
        return len(self.file_classes)

    def __getitem__(self, index):
        _path, _clsid = self.file_classes[index]
        return _path, _clsid


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p2",
        "--wds_target_dir",
        type=str,
        default=wds_target_dir,
        help="path to dataset",
    )
    parser.add_argument(
        "-s", "--split", type=str, default="train", help="split to convert"
    )
    parser.add_argument(
        "-c", "--category_name", type=str, default="cake", help="category name"
    )
    parser.add_argument("--max_size", type=float, default=1, help="gb per shard")
    opt = parser.parse_args()
    os.makedirs(opt.wds_target_dir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    to_tensor = ToTensor()

    if False:
        global_feats = np.random.randn(200, 128)
        np.save(os.path.join(global_feat_path), global_feats)
        print("global_feats saved")
        exit()

    writer = wds.ShardWriter(
        os.path.join(opt.wds_target_dir, "{}-%06d.tar".format(opt.split)),
        maxcount=1e6,
        maxsize=opt.max_size * 1e9 * 0.1,
    )  # -> each shard will be 0.1GB
    # Iterate over the files in the dataset directory

    file_index = 0

    dataset = _Database()
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    for _img_path, _clsid in tqdm(dl, total=len(dl)):
        _img_path = _img_path[0]
        _clsid = _clsid[0]

        wds_dict = {}
        wds_dict["__key__"] = f"{file_index}".zfill(10)

        wds_dict["image.jpg"] = Image.open(_img_path).convert("RGB")
        wds_dict["cls_id.cls"] = int(_clsid) - 1
        assert int(_clsid) - 1 >= 0

        writer.write(wds_dict)
        file_index += 1
        # print("image index: ", file_index)
    writer.close()
    print("done")
