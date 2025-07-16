from einops import repeat
import torch
from tqdm import tqdm
import random
from diffusers import AutoencoderKL
from utils_common import print_rank_0
import wandb
from einops import rearrange
import numpy as np


def out2img(samples):
    return torch.clamp(127.5 * samples + 128.0, 0, 255).to(
        dtype=torch.uint8, device="cuda"
    )


def has_text(args):
    if "uncond" in args.model.name:
        return False
    elif "celebamm" in args.data.name:
        return True
    elif "coco" in args.data.name:
        return True
    else:
        return False


def is_video(args):
    if hasattr(args.model.params, "video_frames"):
        if args.model.params.video_frames > 0:
            return True
        elif args.model.params.video_frames == 0:
            return False
        else:
            raise ValueError("video_frames must be >= 0")
    else:
        return False


def kl_get_vae(args, device):
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    # stabilityai/sd-vae-ft-ema
    vae.eval()
    return vae


def kl_get_encode_decode_fn(args, device):
    if args.is_latent:
        vae = kl_get_vae(args, device)

        if args.use_latent:
            @torch.no_grad()
            def encode_fn(latent):
                return latent
        else:
            @torch.no_grad()
            def encode_fn(img, mini_bs=25):
                img = img / 255.0
                img = img * 2.0 - 1.0

                ############################################################
                for i in range(0, len(img), mini_bs):
                    _img = img[i : i + mini_bs]
                    _feat = vae.encode(_img).latent_dist.sample().mul_(0.18215)
                    if i == 0:
                        feats = _feat
                    else:
                        feats = torch.cat([feats, _feat], dim=0)
                ############################################################
                return feats

        @torch.no_grad()
        def decode_fn(z, mini_bs=25):
            ############################################################
            z = z/0.18215
            for i in range(0, len(z), mini_bs):
                _z = z[i : i + mini_bs]
                x = vae.decode(_z).sample
                if i == 0:
                    samples = x
                else:
                    samples = torch.cat([samples, x], dim=0)
            samples = out2img(samples)
            ############################################################
            return samples

    else:
        raise NotImplementedError("kl_get_encode_decode_fn not implemented")
        def encode_fn(img):
            img = img / 255.0
            img = img * 2.0 - 1.0
            return img

        def decode_fn(img):
            return out2img(img)

    return encode_fn, decode_fn


def kl_get_dynamics(args, device):
    if args.dynamic == "dynsit":
        raise NotImplementedError("dynsit not implemented")
        from dynamics.dynamic_sit import create_transport, Sampler

        transport = create_transport(
            path_type=args.train.path_type,
            prediction=args.train.prediction,
            loss_weight=args.train.loss_weight,
            train_eps=args.train.train_eps,
            sample_eps=args.train.sample_eps,
        )  # default: velocity;
        training_losses_fn = transport.training_losses
        transport_sampler = Sampler(transport)
        sample_fn = transport_sampler.sample_ode(
            num_steps=50
        )  # default to ode sampling
    elif args.dynamic == "fm":
        from dynamics.fm import FlowMatching
        fm = FlowMatching(with_shortcut=False)
        training_losses_fn = fm.training_losses
        _sample_fn = fm.sample_fn
        def sample_fn(z, model_fn, **model_kwargs):
            res =  _sample_fn(z, model_fn, **model_kwargs)
            samples = repeat(res, "b c h w ->7 b c h w")
            return samples
    elif args.dynamic == "meanflow":
        from dynamics.meanflow import MeanFlow
        mf_kwargs = args.shortcut.meanflow
        fm = MeanFlow(**mf_kwargs)
        training_losses_fn = fm.training_losses
        _sample_fn = fm.sample_fn
        def sample_fn(z, model_fn, **model_kwargs):
            res =  _sample_fn(z, model_fn, **model_kwargs)
            samples = repeat(res, "b c h w ->7 b c h w")
            return samples
    elif args.dynamic == "fmshortcut":
        from dynamics.fm import FlowMatching
        fm = FlowMatching(with_shortcut=True)
        training_losses_fn = fm.training_losses
        _sample_fn = fm.sample_fn
        def sample_fn(z, model_fn, **model_kwargs):
            res =  _sample_fn(z, model_fn, **model_kwargs)
            samples = repeat(res, "b c h w ->7 b c h w")
            return samples


    elif args.dynamic == "dyndit":
        raise NotImplementedError("dyndit not implemented")
        from dynamics.dynamic_dit import create_diffusion

        learn_sigma = args.model.params.learn_sigma
        print(f"dit dynamic learn_sigma: {learn_sigma}")
        diffusion_train = create_diffusion(
            timestep_respacing=""
        )  # default: 1000 steps, linear noise schedule
        # create_diffusion(learn_sigma=learn_sigma, timestep_respacing="")
        diffusion_sample = create_diffusion(str(args.dyndit.step_num))

        def training_losses_fn(model, x, model_kwargs=None):
            t = torch.randint(
                0, diffusion_train.num_timesteps, (len(x),), device=device
            )
            loss_dict = diffusion_train.training_losses(model, x, t, model_kwargs)
            return loss_dict

        def sample_fn(z, model_fn, **model_kwargs):  # model_fn
            # Sample images:
            samples = diffusion_sample.p_sample_loop(
                model_fn,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device,
            )  # model.forward_with_cfg
            samples = repeat(samples, "b c h w ->7 b c h w")
            return samples

    elif args.dynamic == "uvitdyn":
        raise NotImplementedError("uvitdyn not implemented")
        from utils.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
        from utils import uvit_sde

        schedule = args.uvitdyn.schedule
        pred = args.uvitdyn.pred
        num_steps = args.uvitdyn.num_steps

        noise_schedule = NoiseScheduleVP(schedule=schedule)
        raise NotImplementedError("uvitdyn not implemented")
        sm = uvit_sde.ScoreModel(nnet=model, pred=pred, sde=uvit_sde.VPSDE())

        def training_losses_fn(model, x, model_kwargs=None):
            loss = uvit_sde.LSimple(sm, x, pred=pred, **model_kwargs)
            d = dict(loss=loss)
            return d

        ############################
        def sample_fn(z, model_fn, **model_kwargs):

            model_fn_final = model_wrapper(
                model_fn,
                noise_schedule,
                time_input_type="0",
                model_kwargs=model_kwargs,
            )
            dpm_solver = DPM_Solver(model_fn_final, noise_schedule)
            x_sampled = dpm_solver.sample(
                z,
                steps=num_steps,
                eps=1e-4,
                adaptive_step_size=False,
                fast_version=True,
            )
            x_sampled = repeat(x_sampled, "b c h w -> ss b c h w", ss=7)
            return x_sampled

    else:
        raise ValueError(f"dynamic={args.dynamic} not supported")

    print(f"using {args.dynamic} dynamics")
    return training_losses_fn, sample_fn


def kl_get_generator(loader, train_steps, accelerator, args, device):

    def get_data_generator(return_cls_id=True):
        _init = train_steps
        for data in tqdm(
            loader,
            disable=not accelerator.is_main_process,
            initial=_init,
            desc="train_steps",
        ):
            if args.use_latent:  
                x = data["latent"].to(device)
                if  args.shortcut.use_repa:
                    #assert "image" in data
                    #image = data["image"].to(device)
                    dino_feature = data["dino_feature"].to(device)
                else:
                    #image = None
                    dino_feature = None
                try:
                    y = data["cls_id"].to(device)
                except:
                    y = None
                if return_cls_id:
                    if args.shortcut.use_repa:
                        yield x, y, dino_feature
                    else:
                        yield x, y
                else:
                    yield x
                
            else:
                raise NotImplementedError("imagenet not implemented")
                if return_cls_id:
                    yield out2img(data["image"]).to(device), data["cls_id"].to(device)
                else:
                    yield out2img(data["image"]).to(device)

    data_generator = get_data_generator()

    real_img_generator = get_data_generator(return_cls_id=False)

    return data_generator, real_img_generator
