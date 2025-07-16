from torchdiffeq import odeint
import torch
import torch.nn.functional as F

import math
import os
from einops import repeat, rearrange
from typing import Any, Dict, List, Optional

from torch.func import jvp
from dynamics.utils_timestep import sample_r_t_logit_normal

def sample_t_fn(x1):
    t = torch.rand(len(x1), device=x1.device)
    t_vector = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return t,t_vector




class MeanFlow:
    def __init__(self,  eps=1e-4, sample_tr="uniform", use_jvp_func=True, loss_weighting=True, weight_c=1e-3, weight_p=1.0, r_eq_t_prob=0.75, ln_mu=-0.4, ln_sigma=1.0):
        self.eps = eps
        self.sample_tr = sample_tr
        self.use_jvp_func = use_jvp_func
        self.loss_weighting = loss_weighting
        self.weight_c = weight_c
        self.weight_p = weight_p
        self.r_eq_t_prob = r_eq_t_prob
        self.ln_mu = ln_mu
        self.ln_sigma = ln_sigma

    def sample_time_pair(self,x1, device, train_progress,verbose=False):
        if self.sample_tr == "uniform":# r is more closer to data, t is more closer to noise
            # Sample from uniform distribution over the triangle
            # First sample from uniform distribution over [0,1]^2
            t = torch.rand(len(x1), device=device)
            t = torch.clamp(t, min=1e-6)
            r = torch.rand(len(x1), device=device)
            r = torch.clamp(r, min=1e-6)
            
            # Transform to get uniform distribution over the triangle
            # This transformation preserves the uniform distribution
            t, r = torch.min(t, r), torch.max(t, r)
        elif self.sample_tr=='anneal':
            r = torch.rand(len(x1), device=device)* max(0.01,train_progress)
            t = torch.rand(len(x1), device=device)* r
        elif self.sample_tr=='anneal_delta':
            max_dt =  max(0.01,train_progress)
            dt =torch.rand(len(x1), device=device)* max_dt
            t = torch.rand(len(x1), device=device)* (1-dt) 
            r = t + dt
        elif self.sample_tr == "log_normal":
            t,r = sample_r_t_logit_normal(x1, mu=self.ln_mu, sigma=self.ln_sigma)
        else:
            raise ValueError(f"sample_tr {self.sample_tr} not supported")
        if verbose:
            for i in range(len(r)):
                print(f"r[{i}]:{r[i]}, t[{i}]:{t[i]}")
        #assert r[0]>=t[0],f"r[0]:{r[0]}, t[0]:{t[0]}"
        r = r.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        if True:
            r_reset =   int(len(r)*self.r_eq_t_prob)
            r[:r_reset] = t[:r_reset]
        return r, t
    
    def fss(self,r,t):
        return t-r
        #return r
    
    def training_losses(self, model, ema_model, x1, sc_kwargs,train_progress,verbose=False, **model_kwargs):
       
        
        
        if False:
            r, t = self.sample_time_pair(x1, x1.device,train_progress=train_progress,verbose=verbose)
            x0 = torch.randn_like(x1,device=x1.device)
            x_t =   (1 - t) * x0 + t * x1
            v_t = x1 - x0
            if True:
                r_tangent = torch.zeros_like(r,device=r.device)
                t_tangent = torch.ones_like(t,device=t.device)
            else:
                r_tangent = torch.ones_like(r,device=r.device)
                t_tangent = torch.zeros_like(t,device=t.device)
            

            def model_fn4jvp(x_t, r, t):
                return model(x_t, t, dt=self.fss(r,t), **model_kwargs)
            if self.use_jvp_func:
                # Convert scalar tangents to tensors with matching shapes
                u_t_pred, dudt = jvp(model_fn4jvp,(x_t,r,t),(v_t,r_tangent,t_tangent))
                if False:
                    print(f"u_t_pred.shape: {u_t_pred.shape}, dudt.shape: {dudt.shape}, v_t.shape: {v_t.shape}, r_tangent.shape: {r_tangent.shape}, t_tangent.shape: {t_tangent.shape},r.shape: {r.shape}, t.shape: {t.shape}")    
            else: 
                u_t_pred = model(x_t, t, dt=self.fss(r,t), **model_kwargs)
                #######
                x_t_ = x_t.clone().requires_grad_(True)
                t_ = t.clone().requires_grad_(True)
                u_theta4grad = model(x_t_, t_, dt=self.fss(r,t_), **model_kwargs)
                # Backward trick version (use_jvp_func=False)
                vdot_scalar = (u_theta4grad * v_t).sum()
                grad_u_wrt_z = torch.autograd.grad(vdot_scalar, x_t_, create_graph=True)[0]
                grad_u_wrt_t = torch.autograd.grad(u_theta4grad.sum(), t_, create_graph=True)[0]
                dudt = grad_u_wrt_z + grad_u_wrt_t
            u_tr_gt = v_t - (t-r)*dudt
            error_all = (u_t_pred-u_tr_gt.detach())**2
            loss_fm = error_all[rearrange(r==t, "b 1 1 1->b")].mean()
            loss_reg = error_all[rearrange(r!=t, "b 1 1 1->b")].mean()
            ret_dict = {"loss_fm": loss_fm, "loss_reg": loss_reg,"train_progress":train_progress,"t_max":t.max(),"t_min":t.min(),"t_mean":t.mean(),"r_max":r.max(),"r_min":r.min(),"r_mean":r.mean(),'r_t_eq':(r==t).sum()/r.numel()}
        else:
            r, t = self.sample_time_pair(x1, x1.device, train_progress=train_progress, verbose=verbose)
            x0 = torch.randn_like(x1,device=x1.device)
            x_t =   (1 - (1-1e-6)*t) * x0 + t * x1
            v_t = x1 - x0
            y = model_kwargs['y']
            
            r_is_t = rearrange(r==t, "b 1 1 1->b")
            r_is_not_t = rearrange(r!=t, "b 1 1 1->b")

            r_flow, r_reg = r[r_is_t], r[r_is_not_t]
            t_flow, t_reg = t[r_is_t], t[r_is_not_t]
            y_flow, y_reg = y[r_is_t], y[r_is_not_t]
            x_t_flow, x_t_reg = x_t[r_is_t], x_t[r_is_not_t]
            v_t_flow, v_t_reg = v_t[r_is_t], v_t[r_is_not_t]

            if True:
                r_tangent = torch.zeros_like(r_reg,device=r.device)
                t_tangent = torch.ones_like(t_reg,device=t.device)
                
           
            def model_fn4jvp(x_t, r, t):
                return ema_model(x_t, t, dt=self.fss(r,t), y=y_reg)
            
            # First, we need to create a function that takes a single tuple argument
            def model_fn_wrapper(*inputs):
                x_t, r, t = inputs
                return ema_model(x_t, t, dt=self.fss(r,t), y=y_reg)

            # Now use F.jvp
            
            
            if len(r_reg)>0:
                if self.use_jvp_func:
                    # Convert scalar tangents to tensors with matching shapes
                    #u_t_pred, dudt = jvp(model_fn4jvp,(x_t_reg,r_reg,t_reg),(v_t_reg ,r_tangent,t_tangent)) 
                    u_t_pred, dudt = torch.autograd.functional.jvp(
                    model_fn_wrapper,
                    (x_t_reg, r_reg, t_reg),
                    (v_t_reg, r_tangent, t_tangent),
                    create_graph=True
                )
                    
                    u_tr_gt_reg = v_t_reg - (t_reg-r_reg)*dudt
                    error_reg = (u_t_pred-u_tr_gt_reg.detach())**2
                else: 
                    #u_t_pred = model(x_t_reg, t_reg, dt=self.fss(r_reg,t_reg), **model_kwargs)
                    #######
                    x_t_ = x_t_reg.clone().requires_grad_(True)
                    t_ = t_reg.clone().requires_grad_(True)
                    u_theta4grad = model(x_t_, t_, dt=self.fss(r_reg,t_reg), **model_kwargs)
                    # Backward trick version (use_jvp_func=False)
                    vdot_scalar = (u_theta4grad * v_t_reg).sum()
                    grad_u_wrt_z = torch.autograd.grad(vdot_scalar, x_t_, create_graph=True)[0]
                    grad_u_wrt_t = torch.autograd.grad(u_theta4grad.sum(), t_, create_graph=True)[0]
                    dudt = grad_u_wrt_z + grad_u_wrt_t
                    u_tr_gt_reg = v_t_reg - (t_reg-r_reg)*dudt
                    error_reg = (u_theta4grad-u_tr_gt_reg.detach())**2
            else:
                error_reg = torch.tensor(0.0,device=x1.device)
            #####
            assert (r_flow==t_flow).sum() == len(r_flow),f"r_flow: {r_flow}, t_flow: {t_flow}"
            u_t_flow = model(x_t_flow, t_flow, dt=self.fss(r_flow,t_flow),y=y_flow)
            error_flow = (u_t_flow-v_t_flow)**2
            
            
            ret_dict = {"loss_fm": error_flow.mean(), "loss_reg": error_reg.mean(),"train_progress":train_progress,"t_max":t.max(),"t_min":t.min(),"t_mean":t.mean(),"r_max":r.max(),"r_min":r.min(),"r_mean":r.mean(),'r_t_eq':(r==t).sum()/r.numel()}
        if not self.loss_weighting:
            loss_reg = error_reg.mean()
            loss_flow = error_flow.mean()
            loss = loss_reg + loss_flow
            ret_dict["loss"] = loss
            return ret_dict
        else:
    
            #loss_reg = error_reg.mean()
            #loss_flow = error_flow.mean()
            #loss = loss_reg + loss_flow
            error_all = torch.cat([error_reg,error_flow],dim=0)
            weight = 1.0 / (error_all.mean() + self.weight_c) ** self.weight_p
            loss = weight.detach() * error_all.mean()
            ret_dict["loss"] = loss
            return ret_dict
       



    @torch.no_grad()
    def sample_fn(self, z, model_fn, step_num, **model_kwargs):
        device = z.device
        verbose = model_kwargs.get("verbose",False)
        if verbose:
            print(f"z.shape: {z.shape}, z.min: {z.min()}, z.max: {z.max()}")
        def sample_fn_wrapper(model,*args,**kwargs):
             res = model(*args,**kwargs)
             if isinstance(res,tuple):#if is repa model, return only the first element
                return res[0]
             else:
                return res
             
        ts = torch.linspace(0,1,step_num+1,device=device)
        ts = repeat(ts, "t->t b 1 1 1", b=len(z))
        
        x_t = z 
        for i in range(step_num):
            t = ts[i]
            r = ts[i+1]
            u = sample_fn_wrapper(model_fn,x_t, t, self.fss(r,t), **model_kwargs)
            #print("z.shape:",z.shape,"r.shape:",r.shape,"t.shape:",t.shape,"u.shape:",u.shape)
            z = z + (t-r)*u
            if verbose:
                print(f"i,x_t: {i},{x_t.shape},{x_t.min()},{x_t.max()}")
        return z


from matplotlib import pyplot as plt
def draw_log_normal_distribution(num_samples, mu=-0.4, sigma=1.0):
    x1 = torch.randn(num_samples)
    t,r = sample_r_t_logit_normal(x1, mu=mu, sigma=sigma)
    plt.hist(t.cpu().numpy(), bins=100, density=True, alpha=0.5, label='t')
    #plt.hist(r.cpu().numpy(), bins=100, density=True, alpha=0.5, label='r')
    plt.legend()
    plt.savefig(f"log_normal_distribution_mu{mu}_sigma{sigma}.png")
            


if __name__ == "__main__":
    if False:
        import torch.nn as nn
        from pathlib import Path
        from copy import deepcopy
        project_path="~/lab/improved_shortcut"
        project_path = Path(project_path).expanduser().resolve()
        import sys
        sys.path.insert(0,str(project_path))
        from models.dit_shortcut import DiT_S_4
        model = DiT_S_4(use_dt_adaptor=True,use_shortcut=True,learn_sigma=False, in_channels=4)
        ema_model = deepcopy(model)
        x1 = torch.randn(10,4,32,32)
        sc_kwargs = {}
        model_kwargs = {}
        fm = MeanFlow(use_jvp_func=True,sample_tr="log_normal")
        _d =fm.training_losses(model, ema_model, x1, sc_kwargs, verbose=True, **model_kwargs)
        print(_d["loss"])
    else:
        draw_log_normal_distribution(100_000, mu=0, sigma=1.0)