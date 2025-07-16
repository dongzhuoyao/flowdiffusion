# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
"""
import random
import shutil
from einops import rearrange, repeat
import numpy as np
from omegaconf import OmegaConf
from dataloader_utils import get_dataloader
import torch
from utils_kl import kl_get_generator
from utils_kl import kl_get_dynamics
from utils_common import get_version_number

from utils.train_utils import (
    create_logger,
    get_latest_checkpoint,
    get_model,
    grad_clip,
    requires_grad,
    update_ema,
    wandb_runid_from_checkpoint,
)
import torch.distributed as dist
from copy import deepcopy
from time import time
import logging
import os
from tqdm import tqdm
import wandb

from utils_kl import kl_get_encode_decode_fn, vis_hyperbolic_interpolate
from utils.my_metrics_offline import MyMetric_Offline as MyMetric


from utils.train_utils_args import rankzero_logging_info
import hydra
from hydra.core.hydra_config import HydraConfig
import accelerate
from wandb_utils import array2grid_pixel, get_max_ckpt_from_dir
import socket
from utils_common import print_rank_0

def update_note(args, accelerator, slurm_job_id):
    args.note = (
        f"kl{get_version_number()}"
        + str(args.note)
        + f"_{args.data.name}"
        + f"_{args.model.name}"
        + f"_{args.dynamic}"
        + f"_bs{args.data.batch_size}"
        + f"_wd{args.optim.wd}"
        + f"_{accelerator.state.num_processes}g"
        + f"_{slurm_job_id}"
    )
    return args.note


def init_z(bs, device, _c, _size, args):

    _zs = torch.randn(
        bs,
        _c,
        _size,
        _size,
        device=device,
    )
    return _zs


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    accelerator = accelerate.Accelerator()
    device = accelerator.device

    accelerate.utils.set_seed(args.global_seed, device_specific=True)
    rank = accelerator.state.process_index

    is_multiprocess = True if accelerator.state.num_processes > 1 else False

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    logging.info(f"slurm_job_id: {slurm_job_id}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.info(args)
        experiment_dir = HydraConfig.get().run.dir
        logging.info(f"Experiment directory created at {experiment_dir}")
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(rank, experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        if args.use_wandb:
            config_dict = OmegaConf.to_container(args, resolve=True)
            config_dict = {
                **config_dict,
                "experiment_dir": experiment_dir,
                "world_size": accelerator.state.num_processes,
                "local_batch_size": args.data.batch_size
                * accelerator.state.num_processes,
                "job_id": slurm_job_id,
            }
            extra_wb_kwargs = dict()
            # if args.ckpt is not None:
            #     runid = wandb_runid_from_checkpoint(args.ckpt)
            #     extra_wb_kwargs["resume"] = "must"
            #     extra_wb_kwargs["id"] = runid
            wandb_name = args.note = update_note(
                args=args, accelerator=accelerator, slurm_job_id=slurm_job_id
            )
            wandb_run = wandb.init(
                project=args.wandb.project,
                name=wandb_name,
                config=config_dict,
                dir=experiment_dir,
                # mode=args.wandb.mode,
                **extra_wb_kwargs,
            )
            wandb_project_url = (
                f"https://wandb.ai/dpose-team/{wandb.run.project}/runs/{wandb.run.id}"
            )
            wandb_sync_command = (
                f"wandb sync {experiment_dir}/wandb/latest-run --append"
            )
    else:
        logger = create_logger(rank)

    model, in_channels, input_size = get_model(args, device)    

    # args.data.sample_fid_bs = min(30, args.data.batch_size // 2)
    args.data.sample_fid_bs = 128
    print_rank_0(f"forced sample_fid_bs to be equal to batch_size={args.data.sample_fid_bs}")
    _fid_eval_batch_nums = args.data.sample_fid_n // (args.data.sample_fid_bs * accelerator.state.num_processes)

    assert _fid_eval_batch_nums > 0, f"{_fid_eval_batch_nums} <= 0"

    ema_model = deepcopy(model).to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.optim.lr, weight_decay=args.optim.wd
    )

    update_ema(
        ema_model, model, decay=0
    )  # Ensure EMA is initialized with synced weights

    training_losses_fn, sample_fn = kl_get_dynamics(args, device)

    _param_amount = sum(p.numel() for p in model.parameters())

    logger.info(f"#parameters: {_param_amount}")

    loader = get_dataloader(args)

    if args.data.name in [
        "cub200_256_cond",
        "pokemon150_256_cond",
        "cub200_256_cond_data_hyper_cub200_balanced_spherical_hierarchy_200d",
        "cub200_256_cond_data_hyper_cub200_flat_300D_c0.1",
    ]:
        embs_data = torch.load(args.data.hyper_feat_path, map_location="cpu")
        class_names, embs = embs_data["objects"], embs_data["embeddings"]
    elif args.data.name == "cub200_256_cond_hyper":
        embs = torch.eye(args.data.num_classes).to(device)
        class_names = None
    else:
        embs = np.eye(args.data.num_classes).astype(np.float32)  # as num_classes is 200

    loader, opt, model, ema_model = accelerator.prepare(loader, opt, model, ema_model)

    if args.ckpt is not None:
        args.ckpt = get_max_ckpt_from_dir(args.ckpt)

    if args.ckpt is not None:  # before accelerate.wrap()
        ckpt_path = args.ckpt
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        # remove module. prefix for 'model' and 'ema'
        # state_dict["model"] = {k.replace("module.", ""): v for k, v in state_dict["model"].items()}
        # state_dict["ema"] = {k.replace("module.", ""): v for k, v in state_dict["ema"].items()}
        model.load_state_dict(state_dict["model"])
        model = model.to(device)
        ema_model.load_state_dict(state_dict["ema"])
        ema_model = ema_model.to(device)
        opt.load_state_dict(state_dict["opt"])

        logging.info("overriding args with checkpoint args")
        logging.info(args)
        train_steps = state_dict["train_steps"]
        best_fid = state_dict["best_fid"]

        logging.info(f"Loaded checkpoint from {ckpt_path}, train_steps={train_steps}")
        requires_grad(ema_model, False)

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema_model.eval()  # EMA model should always be in eval mode

    train_dg, real_img_dg = kl_get_generator(loader, train_steps, accelerator, args, device)

    if args.use_ema:
        print_rank_0("using ema model for sampling...")
        model_fn = accelerator.unwrap_model(ema_model).forward_without_cfg
    else:
        raise ValueError("use_ema must be True")

    encode_fn, decode_fn = kl_get_encode_decode_fn(args, device)

    train_dg, real_img_dg = kl_get_generator(
        loader, train_steps, accelerator, args, device
    )
    def sample_img(bs, args, zs=None):
        if zs is None:
            _zs = init_z(bs, device, in_channels, input_size, args)
        else:
            _zs = zs
        vis_config = dict()
        if args.data.name.startswith("imagenet")
         and args.data.num_classes > 0:
            ys = torch.randint(0, args.data.num_classes - 1, (len(_zs),)).to(device)
            sample_model_kwargs = dict(y=ys)
        else:
            sample_model_kwargs = dict()
        # print_rank_0("sample_model_kwargs: ", sample_model_kwargs)
        ##############
        try:
            samples = sample_fn(_zs, model_fn, **sample_model_kwargs)[-1]
        except Exception as e:
            logging.info("sample_fn error", exc_info=True)
            if accelerator.is_main_process:
                if "sampling_error" not in wandb_run.tags:
                    wandb_run.tags = wandb_run.tags + ("sampling_error",)
                    print_rank_0("sampling_error, wandb_run.tags:", wandb_run.tags)
            samples = torch.rand_like(_zs)

        accelerator.wait_for_everyone()

        samples = decode_fn(samples)
        ys_global = accelerator.gather(ys.contiguous())
        out_sample_global = accelerator.gather(samples.contiguous())
        vis_config["y"] = ys_global
        return out_sample_global, samples, vis_config

    fake_path = args.data.name + "_fid_fake"
    fake_path = os.path.join('./samples', fake_path)
    my_metric = MyMetric(npz_real=args.data.npz_real, fake_path=fake_path)
    # Evaluate
    with torch.no_grad():  # very important
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            my_metric.reset()
 
        ########
        logger.info(
            f"Generating EMA samples, batch size_gpu = {args.data.sample_fid_bs}..."
        )

        vis_wandb_sample = None
        start_time_samplingfid = time()
        for _b_id in tqdm(
            range(_fid_eval_batch_nums),
            desc="sampling FID on the fly",
            total=_fid_eval_batch_nums,
        ):
            out_sample_global, samples, vis_config = sample_img(
                bs=args.data.sample_fid_bs, args=args
            )
            if _b_id == 0:
                vis_wandb_sample = out_sample_global
            # if accelerator.is_main_process:
            my_metric.update_fake_with_vis_config(out_sample_global, vis_config=vis_config, embs=embs)
            del out_sample_global, samples
            torch.cuda.empty_cache()

        ###

        if accelerator.is_main_process and args.use_wandb:
            sample_time_min = (time() - start_time_samplingfid) / 60
            _metric_dict = my_metric.compute()
            # my_metric.reset()
            fid = _metric_dict["fid"]
            best_fid = min(fid, best_fid)
            _metric_dict = {f"eval/{k}": v for k, v in _metric_dict.items()}
            logger.info(f"FID: {fid}, best_fid: {best_fid}")
            wandb_dict = {
                f"best_fid": best_fid,
                "sample_time_min": sample_time_min,
                "vis/sample": wandb.Image(
                    array2grid_pixel(vis_wandb_sample[:16])
                ),
            }
            wandb_dict.update(_metric_dict)
            wandb.log(
                wandb_dict,
                step=train_steps,
            )
        rankzero_logging_info(rank, "Generating EMA samples done.")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()