# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
"""
import random
import shutil
from einops import rearrange, repeat
from omegaconf import OmegaConf
from dataloader_utils import get_dataloader
import torch
from utils_kl import kl_get_generator
from utils_kl import kl_get_dynamics
from utils_common import get_version_number
from accelerate.utils import DistributedDataParallelKwargs

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

from utils_kl import kl_get_encode_decode_fn


from utils.train_utils_args import rankzero_logging_info
import hydra
from hydra.core.hydra_config import HydraConfig
import accelerate
from wandb_utils import array2grid_pixel, get_max_ckpt_from_dir
import socket
from utils_common import print_rank_0



def reset_args_by_cluster(num_processes, args):
    if num_processes > 1:
        if "juwels" in socket.gethostname():
            args.data.num_workers = 1
            print_rank_0(f"juwels node, setting num_workers to {args.data.num_workers}")
        elif "fau.de" in socket.gethostname():
            args.data.num_workers = min(num_processes, 8)
            print_rank_0(f"alex node, setting num_workers to {args.data.num_workers}")
        else:
            print_rank_0(
                f"unknown hostname {socket.gethostname()}, setting num_workers to {args.data.num_workers}"
            )

    # args.data.sample_vis_n = args.data.batch_size
    print_rank_0(f"sample_vis_n set to batch size: {args.data.sample_vis_n}")
    return args


def update_note(args, accelerator, slurm_job_id):
    if args.dynamic == "fm":
        _sc_note = f"{args.dynamic}"
    else:
        _sc_note = (f"{args.dynamic}"
                   f"_{args.shortcut.alg_type}"
                   f"_LR{args.shortcut.loss_reduce}"
                   f"_{args.shortcut.annealing_mode}"
                   f"_repa{int(args.shortcut.use_repa)}w{args.shortcut.repa_w}ep{args.shortcut.encoder_path}"
                   f"_etazero{int(args.shortcut.etazero)}"
                   f"_dtneg{int(args.shortcut.dt_negative)}loss{int(args.shortcut.dt_negative_loss)}"
                   f"_lossAnn{int(args.shortcut.annealing_losses)}"
                   f"_flowlosspyramid{int(args.shortcut.flowloss_pyramid)}"
                   f"_bstbranch{args.shortcut.bootstrap_branch}"
                   f"_bst{args.shortcut.bootstrap_every}w{args.shortcut.bst_weight}")

    args.note = (
        f"kl{get_version_number()}"
        + str(args.note)
        + _sc_note
        + f"_{args.optim.name}bs{args.data.batch_size}iters{args.data.train_steps}"
        #+ f"_wd{args.optim.wd}"
                + f"_{args.data.name}"
        + f"_{args.model.name}"
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


#################################################################################
#                                  Training Loop                                #
#################################################################################


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(args):
    if args.debug:
        args.data.batch_size = 16
        args.ckpt_every = 10
        args.data.sample_fid_n = 1_00
        args.data.sample_fid_bs = 4
        args.data.sample_fid_every = 5
        args.data.sample_vis_every = 3
        args.data.sample_vis_n = 2
        args.shortcut.num_steps = 16
        print_rank_0("debug mode, using smaller batch size and sample size")
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Create DDP kwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # Initialize accelerator with kwargs_handlers
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])

    device = accelerator.device
    accelerate.utils.set_seed(args.global_seed, device_specific=True)
    rank = accelerator.state.process_index
    # args = reset_args_by_cluster(accelerator.state.num_processes, args)
    logging.info(
        f"Starting rank={rank}, world_size={accelerator.state.num_processes}, accelerator.mixed_precision={accelerator.mixed_precision},device={device}."
    )
    is_multiprocess = True if accelerator.state.num_processes > 1 else False

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    logging.info(f"slurm_job_id: {slurm_job_id}")

    train_steps = 0

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
            if args.ckpt is not None:
                if not args.wandb_fork:
                    runid = wandb_runid_from_checkpoint(args.ckpt)
                    extra_wb_kwargs["resume"] = "must"
                    extra_wb_kwargs["id"] = runid
                else:
                    print("wandb fork mode, not resuming run the same wandb id, starting new run") 
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

    if True:
        args.data.sample_fid_bs = min(30, args.data.batch_size // 2)
        # https://github.com/baofff/U-ViT/blob/ce551708dc9cde9818d2af7d84dfadfeb7bd9034/configs/imagenet512_uvit_large.py#L66C9-L66C27
        print_rank_0(
            f"forced sample_fid_bs to be equal to batch_size={args.data.sample_fid_bs}"
        )
        _fid_eval_batch_nums = args.data.sample_fid_n // (
            args.data.sample_fid_bs * accelerator.state.num_processes
        )
        assert _fid_eval_batch_nums > 0, f"{_fid_eval_batch_nums} <= 0"

    ema_model = deepcopy(model).to(device)

    if args.optim.name == "muon":
        #https://github.com/KellerJordan/modded-nanogpt/blob/d700b8724cbda3e7b1e5bcadbc0957f6ad1738fd/train_gpt.py#L515
        from muon import Muon
       # Find ≥2D parameters in the body of the network -- these should be optimized by Muon
        muon_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
        # Find everything else -- these should be optimized by AdamW
        adamw_params = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
        assert len(muon_params) > 0, "no muon params found"
        assert len(adamw_params) > 0, "no adamw params found"
        print_rank_0(f"muon params: {len(muon_params)}, adamw params: {len(adamw_params)}")
        # Create the optimizer
        opt = [Muon(muon_params, lr=args.optim.lr, momentum=0.95, rank=rank, world_size=accelerator.state.num_processes),
                    torch.optim.AdamW(adamw_params, lr=args.optim.lr_adamw, betas=(0.90, 0.95), weight_decay=args.optim.wd_adamw,fused=args.optim.fused_adamw)]
    elif args.optim.name == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.optim.lr, weight_decay=args.optim.wd, fused=args.optim.fused
        )
    elif args.optim.name == "adam":
        opt = torch.optim.Adam(
            model.parameters(), lr=args.optim.lr, weight_decay=args.optim.wd, fused=args.optim.fused
        )
    else:
        raise ValueError(f"Optimizer {args.optim.name} not supported")

    update_ema(
        ema_model, model, decay=0
    ) 

    training_losses_fn, sample_fn = kl_get_dynamics(args, device)

    _param_amount = sum(p.numel() for p in model.parameters())

    logger.info(f"#parameters: {_param_amount}")

    loader = get_dataloader(args)

    loader, opt, model, ema_model = accelerator.prepare(loader, opt, model, ema_model)

    if args.ckpt is not None:
        if os.path.isdir(args.ckpt):
            args.ckpt = get_max_ckpt_from_dir(args.ckpt)
        
            pass 
    if args.ckpt is not None:  # before accelerate.wrap()
        ckpt_path = args.ckpt
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["model"])
        model = model.to(device)
        ema_model.load_state_dict(state_dict["ema"])
        ema_model = ema_model.to(device)
        opt.load_state_dict(state_dict["opt"])

        logging.info("overriding args with checkpoint args")
        logging.info(args)
        train_steps = state_dict["train_steps"]
        

        logging.info(f"Loaded checkpoint from {ckpt_path}, train_steps={train_steps}")
        requires_grad(ema_model, False)
        if rank == 0:
            shutil.copy(ckpt_path, checkpoint_dir)

    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema_model.eval()  # EMA model should always be in eval mode

    log_steps = 0
    running_loss = 0
    start_time = time()

    sample_vis_n = args.data.sample_vis_n
    assert (
        sample_vis_n <= args.data.batch_size // 2
    ), f"{sample_vis_n} > {args.data.batch_size}"

    zs_fixed = init_z(sample_vis_n, device, in_channels, input_size, args)
    rankzero_logging_info(rank, f"zs shape: {zs_fixed.shape}")

    if args.use_ema:
        print_rank_0("using ema model for sampling...")

        model_fn = accelerator.unwrap_model(ema_model).forward_without_cfg

    else:
        raise ValueError("use_ema must be True")

    encode_fn, decode_fn = kl_get_encode_decode_fn(args, device)

    train_dg, real_img_dg = kl_get_generator(
        loader, train_steps, accelerator, args, device
    )

    num_step_list= [1,2,3,4,6,8,12,15,64]
    num_step_bestfid_list = [666]*len(num_step_list)

    def sample_img(bs, args, zs=None, step_num=None):
        if zs is None:
            _zs = init_z(bs, device, in_channels, input_size, args)
        else:
            _zs = zs
        vis_config = dict()
        if  args.data.num_classes > 0 and "cfm" not in args.data.name:
            ys = torch.randint(0, args.data.num_classes - 1, (len(_zs),)).to(device)
            sample_model_kwargs = dict(y=ys)
        elif "cfm" in args.data.name:
            _, y = next(train_dg)
            y = y[:len(_zs)]
            assert len(y) == len(_zs)
            sample_model_kwargs = dict(y=y)
        else:
            sample_model_kwargs = dict()
        if step_num is  None:
            step_num = args.shortcut.num_steps
        sample_model_kwargs["step_num"] = step_num
        #print_rank_0("sample_model_kwargs: ", sample_model_kwargs)
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

        out_sample_global = accelerator.gather(samples.contiguous())
        return out_sample_global, samples, vis_config

    
    from utils.my_metrics_offline import MyMetric_Offline as MyMetric
    my_metric = MyMetric(npz_real=args.data.npz_real)

    if True:
        gt_img = next(real_img_dg)
        print_rank_0("gt_img.shape", gt_img.shape, gt_img.min(), gt_img.max())

        with torch.no_grad():
            gt_decoded = encode_fn(gt_img)
            gt_decoded = decode_fn(gt_decoded)
            print_rank_0(
                "gt_decoded.shape", gt_decoded.shape, gt_decoded.min(), gt_decoded.max()
            )

        gt_img = accelerator.gather(gt_img.contiguous())
        gt_decoded = accelerator.gather(gt_decoded.contiguous())
        if accelerator.is_main_process and args.use_wandb:
            wandb_dict = {
                "vis/gt": wandb.Image(array2grid_pixel(gt_img[:16])),
                "vis/gt_decoded": wandb.Image(array2grid_pixel(gt_decoded[:16])),
            }
            wandb.log(wandb_dict)
            logging.info(wandb_project_url + "\n" + wandb_sync_command)
    

    while train_steps < args.data.train_steps:
        if args.shortcut.use_repa:
            x, y, dino_feature = next(train_dg)
            x = encode_fn(x)
            zs = [dino_feature] # # [bs,256,768]
            model_kwargs = (
                dict(
                    y=y,
                )
                if y is not None
                else dict()
            )
            model_kwargs.update(
                    zs=zs)
        else:
            x, y = next(train_dg)
            x = encode_fn(x)
            model_kwargs = dict(y=y)

        model_kwargs["train_progress"] = train_steps / args.data.train_steps
        sc_kwargs = args.shortcut
        with accelerator.autocast():
            loss_dict = training_losses_fn(model=model, ema_model=ema_model, x1=x, sc_kwargs=sc_kwargs, **model_kwargs)
        loss = loss_dict["loss"].mean()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if isinstance(opt, list):#muon
            for optimizer in opt:
                optimizer.step()
                optimizer.zero_grad()
        else:
            opt.step()
            opt.zero_grad()
        update_ema(ema_model, model)

        running_loss += loss.item()
        log_steps += 1
        train_steps += 1

        if train_steps % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            if is_multiprocess:
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / accelerator.state.num_processes
            if accelerator.is_main_process:
                logging.info(
                    f"(step={train_steps:07d}/{args.data.train_steps}), Train Loss: {avg_loss:.4f}, BS-1GPU: {len(x)} Train Steps/Sec: {steps_per_sec:.2f}, slurm_job_id: {slurm_job_id}, {experiment_dir}"
                )
                logging.info(wandb_sync_command)
                latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
                logging.info(latest_checkpoint)
                logging.info(wandb_project_url)
                logging.info(wandb_name)

                if args.use_wandb:
                    wandb_dict = {
                        "train_loss": avg_loss,
                        "train_steps_per_sec": steps_per_sec,
                        "train_steps": train_steps,
                        "train_steps_accum": train_steps // args.accum,
                        "bs_1gpu": len(x),
                        "param_amount": _param_amount,
                        "datarange/x_min": x.min(),
                        "datarange/x_max": x.max(),
                        "datarange/x_mean": x.mean(),
                        "datarange/x_std": x.std(),
                    }
                    wandb_dict.update(loss_dict)
                    wandb.log(
                        wandb_dict,
                        step=train_steps,
                    )
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

        if train_steps % args.ckpt_every == 0 and train_steps > 0:
            if accelerator.is_main_process:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "train_steps": train_steps,
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                wandb.run.summary["latest_checkpoint_path"] = checkpoint_path
                logging.info(f"Saved checkpoint to {checkpoint_path}")
            accelerator.wait_for_everyone()

        if train_steps % args.data.sample_vis_every == 0 and train_steps > 0:

            zs_random = init_z(sample_vis_n, device, in_channels, input_size, args)
            out_sample_global_random, samples, vis_config = sample_img(
                bs=sample_vis_n, args=args, zs=zs_random
            )
            out_sample_global_fixed, samples, vis_config = sample_img(
                bs=sample_vis_n, args=args, zs=zs_fixed
            )
            
            if accelerator.is_main_process:
                wandb_dict.update(
                    {
                        "vis/sample_fixed": wandb.Image(
                            array2grid_pixel(out_sample_global_fixed[:16])
                        ),
                        "vis/sample_random": wandb.Image(
                            array2grid_pixel(out_sample_global_random[:16])
                        ),
                    }
                )

                wandb.log(
                    wandb_dict,
                    step=train_steps,
                )
                rankzero_logging_info(rank, "Generating samples done.")
            torch.cuda.empty_cache()

        if train_steps % args.data.sample_fid_every == 0 and train_steps > 0:
            for iii, step_num in enumerate(num_step_list):
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
                            bs=args.data.sample_fid_bs, args=args, step_num=step_num
                        )
                        if _b_id == 0:
                            vis_wandb_sample = out_sample_global
                        if accelerator.is_main_process:
                            my_metric.update_fake(out_sample_global)
                        del out_sample_global, samples
                        torch.cuda.empty_cache()

                    ###

                    if accelerator.is_main_process and args.use_wandb:
                        sample_time_min = (time() - start_time_samplingfid) / 60
                        _metric_dict = my_metric.compute()
                        my_metric.reset()
                        fid = _metric_dict["fid"]
                        num_step_bestfid_list[iii] = min(fid, num_step_bestfid_list[iii])
                        _metric_dict = {f"eval/{k}": v for k, v in _metric_dict.items()}
                        logger.info(f"FID: {fid}, best_fid: {num_step_bestfid_list[iii]}")
                        wandb_dict = {
                            f"fidbest/best_fid_{step_num}": num_step_bestfid_list[iii],
                            f"fid/fid_{step_num}": fid,
                            f"sample_time_min_{step_num}": sample_time_min,
                            f"vis/sample_{step_num}": wandb.Image(
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

    model.eval()

    logger.info("Done!")


if __name__ == "__main__":
    main()
