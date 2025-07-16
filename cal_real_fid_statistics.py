import os
from PIL import Image
import torch
from tqdm import tqdm
from einops import rearrange
import hydra
from utils.eval_tools.fid_score import calculate_real_statistics


def out2img(samples):
    return torch.clamp(127.5 * samples + 128.0, 0, 255).to(
        dtype=torch.uint8, device="cuda"
    )


def get_wds_dl(args):
    from dataloader_utils import get_dataloader

    dl = get_dataloader(args)
    return dl


def gen_framefid_stat_video(
    real_img_root,
    dl,
    fid_num=50000,
):
    if not os.path.exists(real_img_root):
        os.makedirs(real_img_root)
    out_path_npy = real_img_root + ".npz"
    idx = 0
    frame_per_batch = 100  # (fid_num // len(dl)) + 2

    def data_generator(dl):
        for videos in tqdm(dl):
            videos = videos["image"]
            b, t, c, h, w = videos.shape
            videos = rearrange(videos, "b t c h w -> (b t) c h w")
            videos = videos[torch.randperm(len(videos))][:frame_per_batch]
            # print(img.shape, img.dtype, img.max(), img.min(), cls_id)
            for _frame in videos:
                yield _frame

    dg = data_generator(dl)

    while True:
        _frame = next(dg)
        _frame = out2img(_frame)
        _frame = _frame.permute(1, 2, 0).cpu().numpy()
        _frame = Image.fromarray(_frame)
        _frame.save(f"{real_img_root}/{idx}.jpg")
        idx += 1
        print(f"Saved {idx} images,{real_img_root}")
        if idx >= fid_num:
            break

    calculate_real_statistics(
        path_real=real_img_root,
        out_path_npy=out_path_npy,
    )
    print(f"fid stat done,{out_path_npy}")


def gen_fid_stat(
    real_img_root,
    dl,
    fid_num=50000,
):
    if not os.path.exists(real_img_root):
        os.makedirs(real_img_root)
    out_path_npy = real_img_root + ".npz"

    def data_generator(dl):
        for _data in dl:
            imgs = _data["image"]
            for img in imgs:
                yield img

    dg = data_generator(dl)
    idx = 0
    while True:
        img = next(dg)
        img = out2img(img)
        img = img.cpu().numpy().transpose(1, 2, 0)
        img = Image.fromarray(img)
        img.save(f"{real_img_root}/{idx}.jpg")
        idx += 1
        print(f"Saved {idx} images,{real_img_root}")
        if idx >= fid_num:
            break
    calculate_real_statistics(
        path_real=real_img_root,
        out_path_npy=out_path_npy,
    )
    print(f"fid stat done,{out_path_npy}")


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(args):
    if True:
        args.data.random_crop = True
        print("as we need to fix size to calculate FID, we set random crop to True")
    dl = get_wds_dl(args)
    shards_path = os.path.dirname(args.data.train_shards_path)
    shards_name = (
        shards_path + "_train" if args.data.subset == "train" else shards_path + "_val"
    )

    real_img_root = "_".join(
        [
            shards_name,
            f"res{args.data.crop_size}",
            "fidstat_real_50k",
        ]
    )
    print(f"real_img_root: {real_img_root}")
    if args.data.video_frames > 0:

        gen_framefid_stat_video(
            dl=dl,
            real_img_root=real_img_root,
            fid_num=50000,
        )
    else:
        gen_fid_stat(
            dl=dl,
            real_img_root=real_img_root,
            fid_num=50000,
        )


if __name__ == "__main__":
    """
    Calculate real FID statistics for different datasets.

    This function processes the given dataset and generates FID statistics.
    It can be run from the command line with different configurations.

    Examples:
        To run with CUB200 dataset:
        $ python cal_real_fid_statistics.py data=cub200_256_cond data.batch_size=200 data.subset=train

        To run with Pokemon dataset:
        $ python cal_real_fid_statistics.py data=faceshq_256_cond data.batch_size=200 data.subset=train

        
    Args:
        args: Command-line arguments parsed by Hydra.
    Returns:
        None
    """
    main()
