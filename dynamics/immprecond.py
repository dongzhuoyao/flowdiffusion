# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
 

import torch
from torch.nn.parallel import DistributedDataParallel


class IMMLoss(torch.nn.Module):
    """Inductive Moment Matching (IMM) Loss
    https://arxiv.org/abs/2503.07565

    Defaults for CIFAR-10 (Table 5)
    """

    def __init__(
        self,
        M=4,  # group size (must be divisible by batch size)
        a=1,  # a = 1 one‑step, a=2 multi‑step/FP16
        b=5,  # shift sigmoid
        k=15,  # eta r mapping fn power
        eta_max=160,  # eta r mapping fn max
        eta_min=0,  # eta r mapping fn min
        eps=1e-8,
    ):
        super().__init__()
        self.M = M
        self.a = a
        self.b = b
        self.k = k
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.eps = eps

    def loss_weight(self, net, t):
        lamb = net.get_logsnr(t)
        # negative time derivative of lamb
        if net.noise_schedule == "vp_cosine":
            dlamb_dt = 2 * torch.pi / (torch.sin(torch.pi * t) + self.eps)
        if net.noise_schedule == "fm":
            dlamb_dt = 2.0 / (t * (1 - t) + self.eps)
        alpha_t, sigma_t = net.get_alpha_sigma(t)
        alpha_t = alpha_t.detach()
        sigma_t = sigma_t.detach()

        weight = (
            0.5
            * torch.sigmoid(self.b - lamb)
            * dlamb_dt
            * (alpha_t**self.a / (alpha_t**2 + sigma_t**2 + self.eps))
        )
        return weight

    def kernel_weight(self, net, t, s):
        if net.f_type == "identity":
            c_out = torch.ones_like(t)
        elif net.f_type == "simple_edm":
            alpha_t, sigma_t = net.get_alpha_sigma(t)
            alpha_s, sigma_s = net.get_alpha_sigma(s)
            c_out = (
                -(alpha_s * sigma_t - alpha_t * sigma_s)
                * (alpha_t**2 + sigma_t**2).rsqrt()
                * net.sigma_data
            )
        elif net.f_type == "euler_fm":
            c_out = -t * net.sigma_data

        weight = 1 / torch.abs(c_out)
        return weight[:, None, None]  # [G, 1, 1]

    def compute_r(self, net, t, s):
        alpha_t, sigma_t = net.get_alpha_sigma(t)

        eta_t = sigma_t / alpha_t
        eps = (self.eta_max - self.eta_min) / 2**self.k
        eta = eta_t - eps

        if net.noise_schedule == "vp_cosine":  # inverse of tan(pi t/2)
            r = (2.0 / torch.pi) * torch.atan(eta)
        elif net.noise_schedule == "fm":  # inverse of t/(1-t)
            r = eta / (1.0 + eta)

        return torch.maximum(s, r)

    def forward(self, dynamic, net, x, class_labels=None):
        B = x.shape[0]
        M = self.M
        assert B % M == 0, f"Batch size ({B}) must be divisible by M ({M})"
        G = B // M

        # sample one (t,s,r) per group
        t_g = torch.rand(G, device=x.device) * (dynamic.T - dynamic.eps) + dynamic.eps
        s_g = torch.rand(G, device=x.device) * (t_g - dynamic.eps) + dynamic.eps
        r_g = self.compute_r(dynamic, t_g, s_g)

        # repeat to per-sample vectors
        t = t_g.repeat_interleave(M).view(B, 1, 1, 1)
        s = s_g.repeat_interleave(M).view(B, 1, 1, 1)
        r = r_g.repeat_interleave(M).view(B, 1, 1, 1)

        noise = torch.randn_like(x) * dynamic.sigma_data
        x_t = dynamic.ddim(noise, x, t=torch.ones_like(t), s=t)
        x_r = dynamic.ddim(x_t, x, t=t, s=r)

        y_t = dynamic(net, x_t, t, s, class_labels=class_labels).view(G, M, -1)  # [G, M, D]
        with torch.no_grad():  # stop grad
            y_r = dynamic(net, x_r, r, s, class_labels=class_labels).view(
                G, M, -1
            )  # [G, M, D]

        # group‑wise loss and kernel weights
        w_l = self.loss_weight(dynamic, t_g)  # [G]
        w_k = self.kernel_weight(dynamic, t_g, s_g)  # [G]

        # pairwise distances & Laplace kernels [G, M, M]
        dist_tt = torch.cdist(y_t, y_t, p=2).clamp(min=self.eps)
        dist_rr = torch.cdist(y_r, y_r, p=2).clamp(min=self.eps)
        dist_tr = torch.cdist(y_t, y_r, p=2).clamp(min=self.eps)

        D = y_t.shape[-1]
        K_tt = torch.exp(-w_k * dist_tt / D)
        K_rr = torch.exp(-w_k * dist_rr / D)
        K_tr = torch.exp(-w_k * dist_tr / D)
        # v‑statistic MMD per group
        mmd_g = (K_tt + K_rr - 2 * K_tr).mean(dim=(1, 2))  # [G]
        return torch.mean(w_l * mmd_g)
    


class IMMPrecond(torch.nn.Module):

    def __init__(
        self,
        img_resolution,  # Image resolution.
        img_channels,  # Number of color channels.
        label_dim=0,  # Number of class labels, 0 = unconditional.
        mixed_precision=None,   
        noise_schedule="fm",   
        sigma_data=0.5, 
        f_type="euler_fm",
        T=0.994,
        eps=0.,  
        temb_type='identity', 
        time_scale=1000.,  
        **model_kwargs,  # Keyword arguments for the underlying model.
    ):
        super().__init__()
 

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.loss_fn = IMMLoss()

        self.label_dim = label_dim
        self.use_mixed_precision = mixed_precision is not None
        if mixed_precision == 'bf16':
            self.mixed_precision = torch.bfloat16
        elif mixed_precision == 'fp16':
            self.mixed_precision = torch.float16 
        elif mixed_precision is None:
            self.mixed_precision = torch.float32
        else:
           raise ValueError(f"Unknown mixed_precision: {mixed_precision}")
            
            
        self.noise_schedule = noise_schedule
 
        self.T = T
        self.eps = eps

        self.sigma_data = sigma_data

        self.f_type = f_type
 
        self.nt_low = self.get_log_nt(torch.tensor(self.eps, dtype=torch.float64)).exp().numpy().item()
        self.nt_high = self.get_log_nt(torch.tensor(self.T, dtype=torch.float64)).exp().numpy().item()
         
    
        
        self.time_scale = time_scale 
         
         
        self.temb_type = temb_type
        
        if self.f_type == 'euler_fm':
            assert self.noise_schedule == 'fm'

    def training_losses(self, model, x, class_labels=None, **model_kwargs):
        return self.loss_fn(dynamic=self, net=model, x=x, class_labels=class_labels, **model_kwargs)
          

    def get_logsnr(self, t):
        dtype = t.dtype
        t = t.to(torch.float64)
        if self.noise_schedule == "vp_cosine":
            logsnr = -2 * torch.log(torch.tan(t * torch.pi * 0.5))
 
        elif self.noise_schedule == "fm":
            logsnr = 2 * ((1 - t).log() - t.log())
            
        logsnr = logsnr.to(dtype)
        return logsnr
    
    def get_log_nt(self, t):
        logsnr_t = self.get_logsnr(t)

        return -0.5 * logsnr_t
    
    def get_alpha_sigma(self, t): 
        if self.noise_schedule == 'fm':
            alpha_t = (1 - t)
            sigma_t = t
        elif self.noise_schedule == 'vp_cosine': 
            alpha_t = torch.cos(t * torch.pi * 0.5)
            sigma_t = torch.sin(t * torch.pi * 0.5)
            
        return alpha_t, sigma_t 

    def add_noise(self, y, t,   noise=None):

        if noise is None:
            noise = torch.randn_like(y) * self.sigma_data

        alpha_t, sigma_t = self.get_alpha_sigma(t)
         
        return alpha_t * y + sigma_t * noise, noise 

    def ddim(self, yt, y, t, s, noise=None):
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_s, sigma_s = self.get_alpha_sigma(s)
        

        if noise is None: 
            ys = (alpha_s -   alpha_t * sigma_s / sigma_t) * y + sigma_s / sigma_t * yt
        else:
            ys = alpha_s * y + sigma_s * noise
        return ys
  
   

    def simple_edm_sample_function(self, yt, y, t, s ):
        alpha_t, sigma_t = self.get_alpha_sigma(t)
        alpha_s, sigma_s = self.get_alpha_sigma(s)
         
        c_skip = (alpha_t * alpha_s + sigma_t * sigma_s) / (alpha_t**2 + sigma_t**2)

        c_out = - (alpha_s * sigma_t - alpha_t * sigma_s) * (alpha_t**2 + sigma_t**2).rsqrt() * self.sigma_data
        
        return c_skip * yt + c_out * y
    
    def euler_fm_sample_function(self, yt, y, t, s ):
        assert self.noise_schedule == 'fm'  

        
        return  yt - (t - s) * self.sigma_data *  y 
          
    def nt_to_t(self, nt):
        dtype = nt.dtype
        nt = nt.to(torch.float64)
        if self.noise_schedule == "vp_cosine":
            t = torch.arctan(nt) / (torch.pi * 0.5) 
 
        elif self.noise_schedule == "fm":
            t = nt / (1 + nt)
            
        t = torch.nan_to_num(t, nan=1)

        t = t.to(dtype)
            

        if (
            self.noise_schedule.startswith("vp")
            and self.noise_schedule == "fm"
            and t.max() > 1
        ):
            raise ValueError(f"t out of range: {t.min().item()}, {t.max().item()}")
        return t

    def get_init_noise(self, shape, device):
        
        noise = torch.randn(shape, device=device) * self.sigma_data
        return noise
    


    def forward_model(
        self,
        model,
        x,
        t,
        s,
        class_labels=None, 
        force_fp32=False,
        **model_kwargs,
    ):
 
              
        
        alpha_t, sigma_t = self.get_alpha_sigma(t)
    
        c_in = (alpha_t ** 2 + sigma_t**2 ).rsqrt() / self.sigma_data  
        if self.temb_type == 'identity': 

            c_noise_t = t  * self.time_scale
            c_noise_s = s  * self.time_scale
            
        elif self.temb_type == 'stride':

            c_noise_t = t * self.time_scale
            c_noise_s = (t - s) * self.time_scale
            
        with torch.amp.autocast('cuda', enabled=self.use_mixed_precision   and not force_fp32, dtype= self.mixed_precision ):
            F_x = model( 
                (c_in * x) ,
                c_noise_t.flatten() ,
                c_noise_s.flatten() ,
                class_labels=class_labels, 
                **model_kwargs,
            )   
        return F_x
    


    
    def forward(
        self,
        model,
        x,
        t,
        s=None, 
        class_labels=None, 
        force_fp32=False, 
        **model_kwargs,
    ):
        dtype = t.dtype  
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        ) 
            
        F_x = self.forward_model(
            model,
            x.to(torch.float32),
            t.to(torch.float32).reshape(-1, 1, 1, 1),
            s.to(torch.float32).reshape(-1, 1, 1, 1) if s is not None else None,
            class_labels, 
            force_fp32,
            **model_kwargs,
        ) 
        F_x = F_x.to(dtype) 
         
        if self.f_type == "identity":
            F_x  =  self.ddim(x, F_x , t, s)  
        elif self.f_type == "simple_edm": 
            F_x = self.simple_edm_sample_function(x, F_x , t, s)   
        elif self.f_type == "euler_fm": 
            F_x = self.euler_fm_sample_function(x, F_x, t, s)  
        else:
            raise NotImplementedError
        return F_x
    


 
    def cfg_forward(
        self,
        model,
        x,
        t,
        s=None, 
        class_labels=None,
        force_fp32=False,
        cfg_scale=None, 
        **model_kwargs,
    ):
        dtype = t.dtype   
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        ) 
        if cfg_scale is not None: 

            x_cfg = torch.cat([x, x], dim=0) 
            class_labels = torch.cat([torch.zeros_like(class_labels), class_labels], dim=0)
        else:
            x_cfg = x 
        F_x = self.forward_model(
            model,
            x_cfg.to(torch.float32),
            t.to(torch.float32).reshape(-1, 1, 1, 1) ,
            s.to(torch.float32).reshape(-1, 1, 1, 1)  if s is not None else None,
            class_labels=class_labels,
            force_fp32=force_fp32,
            **model_kwargs,
        ) 
        F_x = F_x.to(dtype) 
        
        if cfg_scale is not None: 
            uncond_F = F_x[:len(x) ]
            cond_F = F_x[len(x) :] 
            
            F_x = uncond_F + cfg_scale * (cond_F - uncond_F) 
         
        if self.f_type == "identity":
            F_x =  self.ddim(x, F_x, t, s)  
        elif self.f_type == "simple_edm": 
            F_x  = self.simple_edm_sample_function(x, F_x , t, s)   
        elif self.f_type == "euler_fm": 
            F_x = self.euler_fm_sample_function(x, F_x , t, s)  
        else:
            raise NotImplementedError
        return F_x
    
    
    @torch.no_grad()
    def sample_fn(self, model, x, t, s, class_labels=None, force_fp32=False, **model_kwargs):
        return self.pushforward_generator(model, x, t, s, class_labels=class_labels, force_fp32=force_fp32, **model_kwargs)

    
    @torch.no_grad()
    def pushforward_generator(self, net, latents, class_labels=None,  discretization=None, mid_nt=None,  num_steps=None,  cfg_scale=None, ):
        # Time step discretization.
        if discretization == 'uniform':
            t_steps = torch.linspace(net.T, net.eps, num_steps+1, dtype=torch.float64, device=latents.device) 
        elif discretization == 'edm':
            nt_min = net.get_log_nt(torch.as_tensor(net.eps, dtype=torch.float64)).exp().item()
            nt_max = net.get_log_nt(torch.as_tensor(net.T, dtype=torch.float64)).exp().item()
            rho = 7
            step_indices = torch.arange(num_steps+1, dtype=torch.float64, device=latents.device)
            nt_steps = (nt_max ** (1 / rho) + step_indices / (num_steps) * (nt_min ** (1 / rho) - nt_max ** (1 / rho))) ** rho
            t_steps = net.nt_to_t(nt_steps)
        else:
            if mid_nt is None:
                mid_nt = []
            mid_t = [net.nt_to_t(torch.as_tensor(nt)).item() for nt in mid_nt] 
            t_steps = torch.tensor(
                [net.T] + list(mid_t), dtype=torch.float64, device=latents.device
            )    
            # t_0 = T, t_N = 0
            t_steps = torch.cat([t_steps, torch.ones_like(t_steps[:1]) * net.eps])
        
        # Sampling steps
        x = latents.to(torch.float64)  
        
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                    
            x = net.cfg_forward(x, t_cur, t_next, class_labels=class_labels, cfg_scale=cfg_scale   ).to(
                torch.float64
            )     
        return x
    
    @torch.no_grad()
    def restart_generator_fn(self, net, latents, class_labels=None, discretization=None, mid_nt=None,  num_steps=None,  cfg_scale=None ):
        # Time step discretization.
        if discretization == 'uniform':
            t_steps = torch.linspace(net.T, net.eps, num_steps+1, dtype=torch.float64, device=latents.device)[:-1]
        elif discretization == 'edm':
            nt_min = net.get_log_nt(torch.as_tensor(net.eps, dtype=torch.float64)).exp().item()
            nt_max = net.get_log_nt(torch.as_tensor(net.T, dtype=torch.float64)).exp().item()
            rho = 7
            step_indices = torch.arange(num_steps+1, dtype=torch.float64, device=latents.device)
            nt_steps = (nt_max ** (1 / rho) + step_indices / (num_steps) * (nt_min ** (1 / rho) - nt_max ** (1 / rho))) ** rho
            t_steps = net.nt_to_t(nt_steps)[:-1]
        else:
            if mid_nt is None:
                mid_nt = []
            mid_t = [net.nt_to_t(torch.as_tensor(nt)).item() for nt in mid_nt] 
            t_steps = torch.tensor(
                [net.T] + list(mid_t), dtype=torch.float64, device=latents.device
            )     
        # Sampling steps
        x = latents.to(torch.float64)  
        
        for i, t_cur in enumerate(t_steps):  
            x = net.cfg_forward(x, t_cur, torch.ones_like(t_cur) * net.eps, class_labels=class_labels,  cfg_scale=cfg_scale  ).to(
                torch.float64
            )    
                
            if i < len(t_steps) - 1:
                x, _ = net.add_noise(x, t_steps[i+1])
                
        return x

def draw_t_s():
        def compute_r(t, s, eta_max=160, eta_min=0, k=11):
            alpha_t, sigma_t = 1-t, t 

            eta_t = sigma_t / alpha_t
            eps = (eta_max - eta_min) / 2**k
            eta = eta_t - eps

           
            r = eta / (1.0 + eta)

            return torch.maximum(s, r)
        
        import matplotlib.pyplot as plt
        N=30
        t = torch.linspace(0, 1, N)
        print(t)
        s =torch.rand(N)*t
        print(s)
        
        r = compute_r(t, s)
        print(r)
        
        # draw N horitonzal lines, each line with (t,r,s) as a point, thus I can see the change of (t,r,s) from top to down.

       

        # Draw lines and dots
        for _i in range(N):
            print("t,r,s",t[_i],r[_i],s[_i])
            # Draw horizontal line
            plt.axhline(y=_i, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
            # Draw dots
            plt.plot([(t[_i],_i), (s[_i],_i)], 'o', markersize=5, color='black', alpha=0.5)
        plt.savefig("dynamics/imm_t_s.png")

    

if __name__ == "__main__":
    if False:
        import sys,os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.dit_shortcut import DiT_S_4
        model = DiT_S_4(
            input_size=32,
            in_channels=3,
            out_channels=3,
        )
        _dynamic = IMMPrecond(img_resolution=32, img_channels=3, label_dim=0, mixed_precision=None, noise_schedule="fm", sigma_data=0.5, f_type="euler_fm", T=0.994, eps=0., temb_type='identity', time_scale=1000.)
        x = torch.randn(16, 3, 32, 32)
        t = torch.tensor([0.5]*16).reshape(-1, 1, 1, 1)
        s = torch.tensor([0.0]*16).reshape(-1, 1, 1, 1)
        pre = _dynamic(model, x, t, s)
        print(pre.shape)
        loss = _dynamic.training_losses(model, x)
        print(loss)
    else:
        draw_t_s()