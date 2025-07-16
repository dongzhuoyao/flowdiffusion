import torch
import torch.nn.functional as F
import numpy as np
import uuid
import os
import shutil
from torchvision.utils import save_image
import click
from PIL import Image
from tqdm import tqdm
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.eval_tools.fid_score import calculate_fid_given_path_fake


class MyMetric_Offline:
    def __init__(
        self,
        npz_real="./data/imagenet256_raw_wds_train_fidstat_real_50k.npz",
        device="cuda",
        fake_path="/tmp/thu/temp_fake",
    ):
        assert os.path.isfile(npz_real)
        self.npz_real = npz_real
        self.device = device
        self.fake = None
        self.fake_path = fake_path + "/" + str(uuid.uuid4().hex[:6])
        print("creating MyMetric_Offline fake path,", self.fake_path)
        shutil.rmtree(self.fake_path, ignore_errors=True)
        os.makedirs(self.fake_path)
        self.num_fake = 0

    def update_real(self, data, is_real=True):
        pass

    def update_fake(self, data, is_real=False):
        data = data.to(self.device)
        if len(data.shape) == 5:  # if it's a video, we evaluate the frame-level FID
            b, f, c, h, w = data.shape
            data = data.reshape(b * f, c, h, w)
        assert len(data.shape) == 4
        for _data in data:
            unique_id = uuid.uuid4().hex[:6]
            save_image(_data / 255.0, f"{self.fake_path}/{unique_id}.png")
        self.num_fake += len(data)

    def update_fake_with_vis_config(self, data, vis_config=None, embs=None, is_real=False):
        data = data.to(self.device)
        if len(data.shape) == 5:  # if it's a video, we evaluate the frame-level FID
            b, f, c, h, w = data.shape
            data = data.reshape(b * f, c, h, w)
        assert len(data.shape) == 4
        for idx, _data in enumerate(data):
            unique_id = uuid.uuid4().hex[:6]
            if vis_config is not None:
                y_emb = vis_config["y"][idx]
                if type(embs) == torch.Tensor:
                    euc_dist = torch.norm(embs.to(self.device) - y_emb, dim=-1)
                elif type(embs) == np.ndarray:
                    euc_dist = torch.norm(torch.from_numpy(embs).to(self.device) - y_emb, dim=-1)
                else:
                    raise ValueError(f"embs type {type(embs)} not supported")
                class_id = torch.argmin(euc_dist, dim=-1).item()
                save_path = f"{self.fake_path}/{unique_id}_{class_id}.png"
            else:
                save_path = f"{self.fake_path}/{unique_id}.png"
            save_image(_data / 255.0, save_path)
            print(f"saved {save_path}")
        self.num_fake += len(data)

    def update_fake_with_vis_config_latent(self, data, vis_config=None, embs=None, is_real=False):
        data = data.to(self.device)
        if len(data.shape) == 5:  # if it's a video, we evaluate the frame-level FID
            b, f, c, h, w = data.shape
            data = data.reshape(b * f, c, h, w)
        assert len(data.shape) == 4
        for idx, _data in enumerate(data):
            unique_id = uuid.uuid4().hex[:6]
            if vis_config is not None:
                y_emb = vis_config["y"][idx]
                if type(embs) == torch.Tensor:
                    euc_dist = torch.norm(embs.to(self.device) - y_emb, dim=-1)
                elif type(embs) == np.ndarray:
                    euc_dist = torch.norm(torch.from_numpy(embs).to(self.device) - y_emb, dim=-1)
                else:
                    raise ValueError(f"embs type {type(embs)} not supported")
                class_id = torch.argmin(euc_dist, dim=-1).item()
                save_path = f"latent_samples/{self.fake_path}/{unique_id}_{class_id}.pt"
            else:
                save_path = f"latent_samples/{self.fake_path}/{unique_id}.pt"
            dir = os.path.dirname(save_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(_data, save_path)
            print(f"saved {save_path}")
        self.num_fake += len(data)

    def compute(self):
        print("computing metrics by npz file...")
        fid = calculate_fid_given_path_fake(
            path_fake=self.fake_path, npy_real=self.npz_real
        )
        return dict(fid=fid, num_fake=self.num_fake)

    def reset(self):
        shutil.rmtree(self.fake_path, ignore_errors=True)
        os.makedirs(self.fake_path)
        self.num_fake = 0


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, dir_fake):
        self.dir_fake = dir_fake
        self.fake_images = os.listdir(dir_fake)

    def __len__(self):
        return len(self.fake_images)

    def __getitem__(self, idx):
        _img_path = os.path.join(self.dir_fake, self.fake_images[idx])
        _img = Image.open(_img_path)
        _img = np.array(_img)
        _img = torch.from_numpy(_img).to("cuda")
        _img = _img.permute(2, 0, 1)
        return _img


@click.command()
@click.option("--dir_fake", type=str, help="Directory of fake images")
@click.option("--npz_real", type=str, help="Path to the real npz file")
def checkfid(dir_fake, npz_real):
    """
    check fid of fake images in dir_fake
    python utils/my_metrics_offline.py checkfid --dir_fake ./data/temp_fake --npz_real ./data/imagenet256_raw_wds_train_fidstat_real_50k.npz
    """
    fid = calculate_fid_given_path_fake(path_fake=dir_fake, npy_real=npz_real)
    print("fid:", fid)


def ttest():
    _metric = MyMetric_Offline()
    _metric.update_real(
        torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8).to("cuda")
    )
    _metric.update_fake(
        torch.randint(0, 255, (100, 3, 299, 299), dtype=torch.uint8).to("cuda")
    )

    print(_metric.compute())


if __name__ == "__main__":
    checkfid()
