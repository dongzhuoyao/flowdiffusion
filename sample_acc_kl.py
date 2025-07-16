# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""

from einops import rearrange
from omegaconf import OmegaConf
import torch


from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import hydra
from torchvision.utils import save_image
from utils.train_utils import get_model, requires_grad
import accelerate
import wandb
import shutil
import uuid
from utils.eval_tools.fid_score import calculate_fid_given_paths
from utils.my_metrics import calculate_fid_given_paths_my_metric
from utils_kl import (
    kl_get_dynamics,
    kl_get_generator,
    kl_get_encode_decode_fn,
)
from utils_common import get_dataset_id2label, print_rank_0
from dataloader_utils import get_dataloader


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = (
        args.allow_tf32
    )  # True: fast but may lead to some small numerical differences
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id is None:
        slurm_job_id = "local"
    print_rank_0(f"slurm_job_id: {slurm_job_id}")

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(args.global_seed, device_specific=True)
    rank = accelerator.state.process_index
    print_rank_0(
        f"Starting rank={rank}, world_size={accelerator.state.num_processes}, device={device}."
    )

    imagenet_id2label = get_dataset_id2label(args.data.name)

    assert args.ckpt is not None, "Must specify a checkpoint to sample from"
   

    

    def load_model(model, ckpt_path):
        assert ckpt_path is not None, "Must specify a checkpoint to sample from"
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        if True:
            _model_dict = state_dict["ema"]
            # Check for NaN in loaded weights
            for k, v in _model_dict.items():
                if torch.isnan(v).any():
                    print_rank_0(f"NaN found in weight {k}")
        else:
            _model_dict = state_dict['model']
            # Check for NaN in loaded weights
            for k, v in _model_dict.items():
                if torch.isnan(v).any():
                    print_rank_0(f"NaN found in weight {k}")
        _model_dict = {k.replace("module.", ""): v for k, v in _model_dict.items()}
        model.load_state_dict(_model_dict)
        model = model.to(device)
        requires_grad(model, False)
        print_rank_0(f"Loaded checkpoint from {ckpt_path}")
        return model
    

    local_bs = args.offline_sample_local_bs
    args.data.batch_size = local_bs  # used for generating captions,etc.
    print_rank_0("local_bs", local_bs)

    loader = get_dataloader(args)
    model, in_channels, input_size = get_model(args, device)
    model = load_model(model, args.ckpt)
    loader, model = accelerator.prepare(loader, model)
    print_rank_0(f"in_channels={in_channels}, input_size={input_size}")
    if args.ag.use_ag:
        model_wg, in_channels, input_size = get_model(args, device)
        model_wg = load_model(model_wg, args.ag.wg_path)
        model_wg = accelerator.prepare(model_wg)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    global_bs = local_bs * accelerator.state.num_processes
    total_samples = int(math.ceil(args.num_fid_samples / global_bs) * global_bs)
    assert (
        total_samples % accelerator.state.num_processes == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.state.num_processes)
    assert (
        samples_needed_this_gpu % local_bs == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"

    iterations = int(samples_needed_this_gpu // local_bs)
    print_rank_0(
        "samples_needed_this_gpu",
        samples_needed_this_gpu,
        "local_bs",
        local_bs,
        "iterations",
        iterations,
    )

    print_rank_0(
        f"Total number of images that will be sampled: {total_samples} with global_batch_size={global_bs}"
    )

    training_losses_fn, sample_fn = kl_get_dynamics(args, device)
    assert args.cfg_scale >= 0.0, "In almost all cases, cfg_scale be >= 1.0"

    # Create folder to save samples:
    ckpt_string_name = (
        os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    )
    shortcut_desc = f"ag{int(args.ag.use_ag)}fid{args.num_fid_samples}cfg{args.cfg_scale}step{args.step_num}bs{args.offline_sample_local_bs}"
    wandb_name = "_".join(
        [
            #"sample_kl_v2",  # sample ground truth images uniformly over the dataset
            args.note,
            shortcut_desc,
            args.data.name,
            args.model.name,
            ckpt_string_name,
            f"{slurm_job_id}",
        ]
    )

    sample_folder_dir = f"{args.sample_dir}/{wandb_name}"
    

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if args.use_wandb:
            entity = args.wandb.entity
            project = args.wandb.project + "_vis"
            print_rank_0(
                f"Logging to wandb entity={entity}, project={project},rank={rank}"
            )
            config_dict = OmegaConf.to_container(args, resolve=True)
            wandb.init(
                project=project,
                name=wandb_name,
                config=config_dict,
                dir=sample_folder_dir,
                resume="allow",
                mode="online",
            )

            wandb_project_url = (
                f"https://wandb.ai/dpose-team/{wandb.run.project}/runs/{wandb.run.id}"
            )
            wandb_sync_command = (
                f"wandb sync {sample_folder_dir}/wandb/latest-run --append"
            )
            wandb_desc = "\n".join(
                [
                    "*" * 24,
                    str(config_dict),
                    wandb_name,
                    wandb_project_url,
                    wandb_sync_command,
                    "*" * 24,
                ]
            )
        else:
            wandb_project_url = "wandb_project_url_null"
            wandb_sync_command = "wandb_sync_command_null"
            wandb_desc = "wandb_desc_null"
        print_rank_0(f"Saving .png samples at {sample_folder_dir}")

    accelerator.wait_for_everyone()

    pbar = range(iterations)
    pbar = tqdm(pbar, total=iterations, desc="sampling") if rank == 0 else pbar
    total = 0

    _, _generator = kl_get_generator(loader, 0, accelerator, args, device)
    encoder_fn, decode_fn = kl_get_encode_decode_fn(args, device)

    

    sample_img_dir = f"{sample_folder_dir}/samples"
    gt_img_dir = f"{sample_folder_dir}/gts"
    shutil.rmtree(sample_img_dir, ignore_errors=True)
    shutil.rmtree(gt_img_dir, ignore_errors=True)
    os.makedirs(sample_img_dir, exist_ok=True)
    os.makedirs(gt_img_dir, exist_ok=True)

    from utils.my_metrics_offline import MyMetric_Offline as MyMetric

    if rank == 0:
        my_metric = MyMetric(npz_real=args.data.npz_real)
        my_metric.reset()

    

    
    for bs_index in pbar:

        z = torch.randn(
            local_bs,
            in_channels,
            input_size,
            input_size,
            device=device,
        )
        if args.data.num_classes > 0:
            y = torch.randint(0, args.data.num_classes - 1, (local_bs,), device=device)
        else:
            y = None

        if not args.vis_nfe_comparison:
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, step_num=args.step_num,verbose=False)
            print_rank_0("model_kwargs", model_kwargs)
            if not args.ag.use_ag:
                model_fn = model.forward
            else:
                def model_fn(*model_args,**model_kwargs):
                    weak_guidance = model_wg.forward(*model_args,**model_kwargs)
                    strong_guidance = model.forward(*model_args,**model_kwargs)
                    weak_guidance = weak_guidance[0] if isinstance(weak_guidance,tuple) else weak_guidance #compatible to REPA
                    strong_guidance = strong_guidance[0] if isinstance(strong_guidance,tuple) else strong_guidance# compatible to REPA
                    x = weak_guidance + args.cfg_scale * (strong_guidance - weak_guidance)
                    return x

            gts = next(_generator)  # [0,255]

            with torch.no_grad():
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                #print("samples.shape", samples.shape,"samples.min", samples.min(),"samples.max", samples.max())
                if args.is_latent:
                    samples = decode_fn(samples)
            gts = gts[: len(samples)]

            sam_4fid, gts_4fid = samples, gts
            sam_4fid = accelerator.gather(sam_4fid.to(device=device))
            gts_4fid = accelerator.gather(gts_4fid.to(device=device))
        else:
            from collections import defaultdict
            step_num_list = [1,2,3,4,6,8,10,12]
            #step_num_list = [1,2,3,4]
            img_vis_list = defaultdict()
            for _step_num in step_num_list:
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, step_num=_step_num,verbose=True)
                print_rank_0("model_kwargs", model_kwargs)
                if not args.ag.use_ag:
                    model_fn = model.forward
                else:
                    def model_fn(*model_args,**model_kwargs):
                        weak_guidance = model_wg.forward(*model_args,**model_kwargs)
                        strong_guidance = model.forward(*model_args,**model_kwargs)
                        weak_guidance = weak_guidance[0] if isinstance(weak_guidance,tuple) else weak_guidance #compatible to REPA
                        strong_guidance = strong_guidance[0] if isinstance(strong_guidance,tuple) else strong_guidance# compatible to REPA
                        x = weak_guidance + args.cfg_scale * (strong_guidance - weak_guidance)
                        return x

                gts = next(_generator)  # [0,255]

                with torch.no_grad():
                    samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                    #print("samples.shape", samples.shape,"samples.min", samples.min(),"samples.max", samples.max())
                    if args.is_latent:
                        samples = decode_fn(samples)
                gts = gts[: len(samples)]

                sam_4fid, gts_4fid = samples, gts
                sam_4fid = accelerator.gather(sam_4fid.to(device=device))
                gts_4fid = accelerator.gather(gts_4fid.to(device=device))
                img_vis_list[_step_num]=sam_4fid#[B,C,H,W]

            if rank == 0:
                
                for img_id in range(len(sam_4fid)):
                    img_vis_various_nfe = []
                    for _step_num in step_num_list:
                        img_vis_various_nfe.append(img_vis_list[_step_num][img_id])
                    img_vis_various_nfe_wandb = torch.concat(img_vis_various_nfe, dim=2)  # This would give [N,C,W,H]

                    print("img_vis_various_nfe_wandb.shape", img_vis_various_nfe_wandb.shape)
                    wandb.log({f"vis/samples_various_nfe_img{img_id}": wandb.Image(img_vis_various_nfe_wandb)})

            print("finished sampling,exit")
            exit()

        print_rank_0(
            "gather done",
            "len(sam_4fid)",
            len(sam_4fid),
            "len(gts_4fid)",
            len(gts_4fid),
        )
        print("sam_4fid.shape", sam_4fid.shape,"sam_4fid.min", sam_4fid.min(),"sam_4fid.max", sam_4fid.max())
        accelerator.wait_for_everyone()
        if rank == 0:
            my_metric.update_fake(sam_4fid)
            if False: #save time 
                # Save samples to disk as individual .png files
                for _iii, sample in enumerate(sam_4fid):
                    unique_id = uuid.uuid4().hex[:6]
                    save_image(sample / 255.0, f"{sample_img_dir}/{unique_id}.png")
                for _iii, sample in enumerate(gts_4fid):
                    unique_id = uuid.uuid4().hex[:6]
                    save_image(sample / 255.0, f"{gt_img_dir}/{unique_id}.png")

            if args.use_wandb and bs_index <= 1:

                captions_sample = [
                    imagenet_id2label[y[_].item()]
                    for _ in range(min(16, len(sam_4fid)))
                ]
                wandb.log(
                    {
                        f"vis/samples_single": [
                            wandb.Image(sam_4fid[i], caption=captions_sample[i])
                            for i in range(min(16, len(sam_4fid)))
                        ],
                        f"vis/gts_single": [
                            wandb.Image(gts_4fid[i])
                            for i in range(min(16, len(gts_4fid)))
                        ],
                    },
                    step=bs_index,
                )

                wandb.log(
                    {
                        f"vis/samples": wandb.Image(sam_4fid[:16]),
                        f"vis/gts": wandb.Image(gts_4fid[:16]),
                    },
                    step=bs_index,
                )
                print_rank_0("log_image into wandb")
            total += global_bs
            if bs_index >= 3 and args.sample_debug:
                print_rank_0("sample_debug, break at bs_index", bs_index)
                break

        accelerator.wait_for_everyone()
    if rank == 0:
        if False:
            fid = calculate_fid_given_paths([sample_img_dir, gt_img_dir])
            print_rank_0("fid", fid)
            mymetric_dict = calculate_fid_given_paths_my_metric(
                sample_img_dir=sample_img_dir, gt_img_dir=gt_img_dir
            )  # calculate other metrics for further confirmation
            mymetric_dict = {f"torchmetrics/{k}": v for k, v in mymetric_dict.items()}
            mymetric_dict.update({"fid_from_uvit_calculation": fid})
        _metric_dict = my_metric.compute()
        my_metric.reset()
        fid = _metric_dict["fid"]
        print_rank_0("fid", fid)
        print("final fid", fid)
        wandb.log({'fid': fid})
        try:
            shutil.rmtree(sample_img_dir)
            shutil.rmtree(gt_img_dir)
            print_rank_0(
                "removed sample_img_dir and gt_img_dir\n",
                sample_img_dir,
                "\n",
                gt_img_dir,
            )
        except Exception as e:
            print_rank_0(f"Error removing directory {sample_img_dir},{gt_img_dir}: {e}")

    print_rank_0("done sampling")
    # accelerator.wait_for_everyone(), important! remove this.


if __name__ == "__main__":

    main()
