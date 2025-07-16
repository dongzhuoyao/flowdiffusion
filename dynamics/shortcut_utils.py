import torch
from torch import Tensor
import numpy as np
import math
from typing import Optional
import torch.nn as nn
import os
from einops import rearrange
import matplotlib.pyplot as plt
eps = 1e-5

def compute_xt_fn(x0: Tensor, x1: Tensor, t: Tensor, eps: float = 1e-5):
    if t.ndim == 1:
        t = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return (1 - (1-eps) * t)*x0 + t*x1


def compute_ut_fn(x0: Tensor, x1: Tensor, t: Tensor):
    return x1 - x0


import torch
import torch.distributions as dist

def sample_beta_batch(B, device,alpha=0.5, beta=0.5):
    """
    Sample from Beta distribution for a batch of size B
    
    Args:
        B (int): Batch size
        alpha (float): First
    """
    beta_dist = dist.Beta(alpha, beta)
    return beta_dist.sample((B,)).to(device)


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


def annealing_losses_fn(loss_flow, loss_bootstrap, loss, train_progress, annealing_losses, loss_reduce=None,bst_weight=None):
            #if annealing_losses:
            #    loss= (1-train_progress)*loss_flow.mean() + train_progress*loss_bootstrap.mean()
            #else:
            #    loss = loss_flow.mean() + loss_bootstrap.mean()
            
            if loss_reduce == "add":
                if annealing_losses:
                    loss = (1-train_progress)*loss_flow.mean() + train_progress*loss_bootstrap.mean()*bst_weight
                else:
                    loss = loss_flow.mean() + loss_bootstrap.mean()*bst_weight
            elif loss_reduce == "merge":
                assert annealing_losses == False, "annealing_losses must be False for merge"
                assert bst_weight ==1, "bst_weight must be 1 for merge"
                loss = loss.mean()
            elif loss_reduce=="huber_weighting":
                weight_c=1e-3
                weight_p=1.0
                weight = 1.0 / (loss.mean() + weight_c) ** weight_p
                loss = weight.detach() * loss.mean()
            else:
                raise ValueError(f"loss_reduce={loss_reduce} is not valid")
            return loss

def annealing_bootstrap_every_fn(bs, train_progress, bootstrap_every, annealing_bootstrap_every):
            if annealing_bootstrap_every==True:
                bootstrap_size = int(bs*train_progress)
                bootstrap_size = max(1, bootstrap_size)
                bootstrap_size= min(bootstrap_size, bs-1)
            elif bootstrap_every > 1:
                bootstrap_size = bs // bootstrap_every
            elif bootstrap_every <1:
                assert bootstrap_every > 0, "bootstrap_every must be greater than 0"
                bootstrap_size = bs - int(bs*bootstrap_every)
            else:
                raise ValueError(f"bootstrap_every={bootstrap_every} is not valid")
            return bootstrap_size
    

def forward_wrapper(model,x,t,dt,y, **kwargs):
    result  = model(x=x, t=t, dt=dt, y=y, **kwargs)
    if kwargs["use_repa"] == True:
        assert isinstance(result, tuple)
        x, zs = result
        return x, zs
    else:
        return result, None 

def shortcut_loss( model, ema_model, x1, y, sc_kwargs, train_progress, verbose=False, **model_kwargs):

        bs, device, dtype = len(x1), x1.device, x1.dtype
        x0 = torch.randn_like(x1,device=device,dtype=dtype)
        ####    
        num_steps = sc_kwargs['num_steps']
        bootstrap_every = sc_kwargs['bootstrap_every']
        alg_type = sc_kwargs['alg_type']
        use_ema_for_bootstrap = sc_kwargs['use_ema_for_bootstrap']
        weighting_mode = sc_kwargs['weighting_mode']
        annealing_mode = sc_kwargs['annealing_mode']
        annealing_losses = sc_kwargs["annealing_losses"]
        annealing_bootstrap_every= sc_kwargs["annealing_bootstrap_every"]
        dt_negative= sc_kwargs["dt_negative"]
        dt_negative_loss = sc_kwargs["dt_negative_loss"]
        loss_reduce= sc_kwargs["loss_reduce"]
        bst_weight = sc_kwargs["bst_weight"]
        use_repa = sc_kwargs["use_repa"]
          
        assert alg_type in ["shortcut","shortcut_baseline", "immfm2","imm_official", "fm"], "alg_type must be one of ['shortcut', 'immfm2', 'fm']"

        if alg_type=="immfm":
            raise NotImplementedError("immfm is not implemented yet")
            eps = 1e-5       
            bst_size = annealing_bootstrap_every_fn(bs=bs, train_progress=train_progress, bootstrap_every=bootstrap_every, annealing_bootstrap_every=annealing_bootstrap_every)
            t,  twodt, twodt_base,dt, dt_base = sample_delta(batch_size=bs, time_steps=num_steps, device=device, annealing_mode=annealing_mode,  train_progress=train_progress, dt_negative=False, verbose=verbose)
            # makes t the same dimension as x0 and x1
            x0_bst = x0[:bst_size]
            x1_bst = x1[:bst_size]
            t_bst = t[:bst_size]
            dt_bst = dt[:bst_size]
            y_bst = y[:bst_size]
            ######
            x0_flow = x0[bst_size:]
            x1_flow = x1[bst_size:]
            t_flow = t[bst_size:]
            dt_flow = dt[bst_size:]
            y_flow = y[bst_size:]
            
            if True:
                t_vec = t_bst.view(-1, *[1]*(x0_bst.ndim-1))
                dt_vec = dt_bst.view(-1, *[1]*(x0_bst.ndim-1))
                
                if verbose:
                    print("x0_bst.shape", x0_bst.shape, "x1_bst.shape", x1_bst.shape,"t_bst.shape", t_bst.shape,"dt_bst.shape", dt_bst.shape,"y_bst.shape", y_bst.shape)
                x_t = (1 - (1-eps) * t_vec)*x0_bst + t_vec*x1_bst 
                t_prime_vec = t_vec + dt_vec
                t_prime = t_bst + dt_bst 
                x_t_prime = (1 - (1-eps) * t_prime_vec)*x0_bst + t_prime_vec*x1_bst 
                
                if False:
                    v_teacher = ema_model(x=x_t, t=t_bst, dt=dt_bst, y=y_bst, **sc_kwargs)
                    v_teacher = v_teacher.detach()
                    v_student = model(x=x_t_prime, t=t_prime,  dt=dt_bst, y=y_bst, **sc_kwargs)
                else:
                    v_teacher = ema_model(x=x_t_prime, t=t_prime, dt=dt_bst, y=y_bst, **sc_kwargs)
                    v_teacher = v_teacher.detach()
                    v_student = model(x=x_t, t=t_bst, dt=dt_bst, y=y_bst, **sc_kwargs)
                bootstrap_loss = (v_student - v_teacher).square().mean()
                if dt_negative:
                    raise NotImplementedError("dt_negative is not implemented yet, need to double-check")
                    assert dt_bst>=0
                    if random.random() < 0.5:
                        neg_tea_x_t, neg_tea_t, neg_tea_dt = x_t, t_bst, dt_bst
                        neg_stu_x_t, neg_stu_t, neg_stu_dt = x_t_prime, t_prime, -dt_bst
                    else:
                        neg_tea_x_t, neg_tea_t, neg_tea_dt = x_t_prime, t_prime, -dt_bst
                        neg_stu_x_t, neg_stu_t, neg_stu_dt = x_t, t_bst, dt_bst

                    neg_v_teacher = ema_model(x=neg_tea_x_t, t=neg_tea_t, dt=neg_tea_dt, y=y_bst, **sc_kwargs)
                    neg_v_student = model(x=neg_stu_x_t, t=neg_stu_t, dt=neg_stu_dt, y=y_bst, **sc_kwargs)
                    neg_reg_loss = (neg_v_student - neg_v_teacher).square().mean()  
                else:
                    neg_reg_loss = 0.0
            if True:
                t_vec = t_flow.view(-1, *[1]*(x0_flow.ndim-1))
                x_t = (1 - (1-eps) * t_vec)*x0_flow + t_vec*x1_flow 
                vf_pred = model(x=x_t, t=t_flow, dt=dt_flow, y=y_flow, **sc_kwargs)
                flow_loss = (vf_pred - (x1_flow - x0_flow)).square().mean()
            
            loss = annealing_losses_fn(loss_flow=flow_loss, loss_bootstrap=bootstrap_loss, train_progress=train_progress, annealing_losses=annealing_losses, loss_reduce=loss_reduce,bst_weight=bst_weight)
            loss = loss + neg_reg_loss
            return {
                    "loss": loss,
                    "bootstrap_loss": bootstrap_loss.detach().mean(),
                    "flow_loss": loss.detach().mean(),
                    "dt_flow_min": dt_flow.min().item(),
                    "dt_flow_max": dt_flow.max().item(),
                    "dt_flow_mean": dt_flow.mean().item(),
                    "dt_bst_min": dt_bst.min().item(),
                    "dt_bst_max": dt_bst.max().item(),
                    "dt_bst_mean": dt_bst.mean().item(),
                    "bootstrap_size": bst_size,
                    "progress": train_progress,
                }
        
        
        elif alg_type=="shortcut_baseline":
            assert bootstrap_every > 0, "bootstrap_every must be greater than 0 for shortcut"
            bst_size = annealing_bootstrap_every_fn(bs=bs, train_progress=train_progress, bootstrap_every=bootstrap_every, annealing_bootstrap_every=annealing_bootstrap_every)
            
            flow_size = bs - bst_size
            if verbose:
                print(f"bs: {bs}, bst_size: {bst_size}, flow_size: {flow_size}, train_progress: {train_progress}, bootstrap_every: {bootstrap_every}, annealing_bootstrap_every: {annealing_bootstrap_every}")
            
            assert annealing_mode=='none'
            xt_flow, t_flow, dt_flow, y_flow, ut_flow = _flow_branch_shortcut_baseline(
                    x1=x1[:flow_size], x0=x0[:flow_size], y=None if y is None else y[:flow_size],  sc_kwargs=sc_kwargs, train_progress=train_progress, verbose=verbose)
            bootstrap_model = ema_model
            xt_bst, t_bst, dt_bst, y_bst, ut_bst,x_t_prime_bst = _bootstrap_branch(
                x1[flow_size:], x0[flow_size:], y=None if y is None else y[flow_size:], bootstrap_model=bootstrap_model, train_progress=train_progress,sc_kwargs=sc_kwargs, verbose=verbose)
            if verbose:
                for i in range(bst_size):
                    print(f"t_bst[{i}]={t_bst[i]}, dt_bst[{i}]={dt_bst[i]}")

            #########################################################
            bst_size = len(dt_bst)
            xt = torch.cat([xt_flow, xt_bst], dim=0).detach()
            t = torch.cat([t_flow, t_bst], dim=0).detach()
            y = None if y is None else torch.cat([y_flow, y_bst], dim=0).detach()
            dt = torch.cat([dt_flow, dt_bst], dim=0).detach()
            ut = torch.cat([ut_flow, ut_bst], dim=0).detach()
            vt,zs_tilde = forward_wrapper(model,x=xt,t=t,dt=dt,y=y, **sc_kwargs)
            loss = (vt - ut)**2
            _loss_flow = loss[:flow_size]
            _loss_bootstrap = loss[flow_size:]
            _loss_bootstrap_neg = torch.tensor(0.0)
            loss = annealing_losses_fn(loss_flow=_loss_flow, loss_bootstrap=_loss_bootstrap,loss=loss, train_progress=train_progress, annealing_losses=annealing_losses, loss_reduce=loss_reduce,bst_weight=bst_weight)

            loss_dict = {
                "loss": loss,
                "bootstrap_loss": _loss_bootstrap.detach().mean(),
                "bootstrap_loss_std": _loss_bootstrap.detach().std(),
                "flow_loss":      _loss_flow.detach().mean(),
                "flow_loss_std": _loss_flow.detach().std(),
                "total_loss":     loss.detach().mean(),
                "dt_mean":   dt.to(torch.float32).mean().item(),
                "dt_max":         dt.max().item(),
                "dt_min":         dt.min().item(),
                "bst_weight":     bst_weight,
                "t_flow_mean":   t_flow.to(torch.float32).mean().item(),
                "t_flow_max":         t_flow.max().item(),
                "t_flow_min":         t_flow.min().item(),
                "t_bst_mean":   t_bst.to(torch.float32).mean().item(),
                "t_bst_max":         t_bst.max().item(),
                "t_bst_min":         t_bst.min().item(),
                "dt_bst_mean":   dt_bst.to(torch.float32).mean().item(),
                "dt_bst_max":         dt_bst.max().item(),
                "dt_bst_min":         dt_bst.min().item(),
                "dt_flow_mean":   dt_flow.to(torch.float32).mean().item(),
                "dt_flow_max":         dt_flow.max().item(),
                "dt_flow_min":         dt_flow.min().item(),
                "bootstrap_size": bst_size,
                "progress": train_progress,
            }
            return  loss_dict
        
        elif alg_type=="immfm2":
           
            eps = 1e-5       
            bst_size = annealing_bootstrap_every_fn(bs=bs, train_progress=train_progress, bootstrap_every=bootstrap_every, annealing_bootstrap_every=annealing_bootstrap_every)
            t,dm,dn = sample_immfm_dmdn(batch_size=bs,  device=device, annealing_mode=annealing_mode,  train_progress=train_progress, dt_negative=dt_negative, num_steps=num_steps, verbose=verbose)
            # makes t the same dimension as x0 and x1
            x0_bst = x0[:bst_size]
            x1_bst = x1[:bst_size]
            t_bst = t[:bst_size]
            dm_bst = dm[:bst_size]
            dn_bst = dn[:bst_size]
            y_bst = y[:bst_size] if y is not None else None
            #########################################################
            x0_flow = x0[bst_size:]
            x1_flow = x1[bst_size:]
            y_flow = y[bst_size:] if y is not None else None
            
            if True:
                t_vec = t_bst.view(-1, *[1]*(x0_bst.ndim-1))
                dm_vec = dm_bst.view(-1, *[1]*(x0_bst.ndim-1))
                dn_vec = dn_bst.view(-1, *[1]*(x0_bst.ndim-1))
                
                
                x_t = (1 - (1-eps) * t_vec)*x0_bst + t_vec*x1_bst 
                t_plus_dm= t_vec + dm_vec
                x_t_plus_dm = (1 - (1-eps) * t_plus_dm)*x0_bst + t_plus_dm*x1_bst #buggy?
                if verbose:
                    print("x0_bst.shape", x0_bst.shape, "x1_bst.shape", x1_bst.shape,"t_bst.shape", t_bst.shape,"dm_bst.shape", dm_bst.shape,"dn_bst.shape", dn_bst.shape,"t_plus_dm.shape", t_plus_dm.shape,"x_t_plus_dm.shape", x_t_plus_dm.shape)
                
                v_teacher = ema_model(x=x_t_plus_dm, t=t_plus_dm, dt=dn_vec, y=y_bst, **sc_kwargs)
                v_teacher = (x_t_plus_dm + dn_vec*v_teacher).detach()
                #####
                v_student = model(x=x_t, t=t_vec, dt=dm_vec+dn_vec, y=y_bst, **sc_kwargs)#[t, t+dm,t+dm+dn]
                v_student = x_t + (dm_vec+dn_vec)*v_student
                bootstrap_loss = (v_student - v_teacher).square().mean()
                if dt_negative:
                    raise NotImplementedError("dt_negative is not implemented yet, need to double-check")
                    neg_v_teacher = ema_model(x=x_t, t=rearrange(t_vec, "b 1 1 1 -> b"), dt=rearrange(dm_vec, "b 1 1 1 -> b"), y=y_bst, **sc_kwargs).detach()
                    neg_v_student = model(x=x_t_plus_dm, t=rearrange(t_plus_dm, "b 1 1 1 -> b"), dt=rearrange(-dm_vec, "b 1 1 1 -> b"), y=y_bst, **sc_kwargs)
                    neg_reg_loss = (neg_v_student + neg_v_teacher).square().mean()  
                else:
                    neg_reg_loss = torch.tensor(0.0)

            if True:
                #dt_flow_mini=torch.ones(len(x0_flow), device=device, dtype=torch.float32)*1.0/num_steps
                dt_flow_mini = torch.zeros(len(x0_flow), device=device, dtype=torch.float32)
                t_flow = torch.rand(len(x0_flow), device=device, dtype=torch.float32)
                t_vec = t_flow.view(-1, *[1]*(x0_flow.ndim-1))
                assert t_vec.min()>=0 and t_vec.max()<=1, f"t_vec.min()={t_vec.min()}, t_vec.max()={t_vec.max()}"
                x_t = (1 - (1-eps) * t_vec)*x0_flow + t_vec*x1_flow 
                vf_pred = model(x=x_t, t=t_flow, dt=dt_flow_mini, y=y_flow, **sc_kwargs)
                flow_loss = (vf_pred - (x1_flow - x0_flow)).square().mean()
            
            loss = bootstrap_loss + flow_loss
            loss = annealing_losses_fn(loss_flow=flow_loss, loss_bootstrap=bootstrap_loss,loss=loss, train_progress=train_progress, annealing_losses=annealing_losses, loss_reduce=loss_reduce,bst_weight=bst_weight)
            loss = loss + neg_reg_loss
            return {
                    "loss": loss,
                    "bootstrap_loss": bootstrap_loss.detach().mean(),
                    "flow_loss": loss.detach().mean(),
                    "neg_reg_loss": neg_reg_loss.detach().mean(),
                    "t_flow_min": t_flow.min().item(),
                    "t_flow_max": t_flow.max().item(),
                    "t_flow_mean": t_flow.mean().item(),
                    "t_bst_min": t_bst.min().item(),
                    "t_bst_max": t_bst.max().item(),
                    "t_bst_mean": t_bst.mean().item(),
                    "dm_min": dm_vec.min().item(),
                    "dm_max": dm_vec.max().item(),
                    "dm_mean": dm_vec.mean().item(),
                    "dn_min": dn_vec.min().item(),
                    "dn_max": dn_vec.max().item(),
                    "dn_mean": dn_vec.mean().item(),
                    "bootstrap_size": bst_size,
                    "progress": train_progress,
                }
        
        elif alg_type=="imm_official":
            imm_k = sc_kwargs["imm_k"]
           
            eps = 1e-5       
            bst_size = annealing_bootstrap_every_fn(bs=bs, train_progress=train_progress, bootstrap_every=bootstrap_every, annealing_bootstrap_every=annealing_bootstrap_every)
            t,r,s = sample_imm_trs(batch_size=bs,  device=device, verbose=verbose, imm_k=imm_k)
            # makes t the same dimension as x0 and x1
            x0_bst = x0[:bst_size]
            x1_bst = x1[:bst_size]
            t_bst = t[:bst_size]
            r_bst = r[:bst_size]
            s_bst = s[:bst_size]
            y_bst = y[:bst_size] if y is not None else None
            #########################################################
            x0_flow = x0[bst_size:]
            x1_flow = x1[bst_size:]
            y_flow = y[bst_size:] if y is not None else None
            
            if True:
                t_vec = t_bst.view(-1, *[1]*(x0_bst.ndim-1))
                r_vec = r_bst.view(-1, *[1]*(x0_bst.ndim-1))
                s_vec = s_bst.view(-1, *[1]*(x0_bst.ndim-1))
                
                
                x_t = (1 - (1-eps) * t_vec)*x0_bst + t_vec*x1_bst 
                x_r = (1 - (1-eps) * r_vec)*x0_bst + r_vec*x1_bst 
                if verbose:
                    print("x0_bst.shape", x0_bst.shape, "x1_bst.shape", x1_bst.shape,"t_bst.shape", t_bst.shape,"r_bst.shape", r_bst.shape,"s_bst.shape", s_bst.shape,"x_r.shape", x_r.shape)
                
                v_teacher = ema_model(x=x_r, t=r_vec, dt=s_vec-r_vec, y=y_bst, **sc_kwargs)
                v_teacher = (x_r + (s_vec-r_vec)*v_teacher).detach()
                #####
                v_student = model(x=x_t, t=t_vec, dt=s_vec - t_vec, y=y_bst, **sc_kwargs)#[t, t+dm,t+dm+dn]
                v_student = x_t + (s_vec - t_vec)*v_student
                bootstrap_loss = (v_student - v_teacher).square().mean()
                

            if True:
                dt_flow_mini = torch.zeros(len(x0_flow), device=device, dtype=torch.float32)
                t_flow = torch.rand(len(x0_flow), device=device, dtype=torch.float32)
                t_vec = t_flow.view(-1, *[1]*(x0_flow.ndim-1))
                assert t_vec.min()>=0 and t_vec.max()<=1, f"t_vec.min()={t_vec.min()}, t_vec.max()={t_vec.max()}"
                x_t = (1 - (1-eps) * t_vec)*x0_flow + t_vec*x1_flow 
                vf_pred = model(x=x_t, t=t_flow, dt=dt_flow_mini, y=y_flow, **sc_kwargs)
                flow_loss = (vf_pred - (x1_flow - x0_flow)).square().mean()
            
            loss = bootstrap_loss + flow_loss
            loss = annealing_losses_fn(loss_flow=flow_loss, loss_bootstrap=bootstrap_loss,loss=loss, train_progress=train_progress, annealing_losses=annealing_losses, loss_reduce=loss_reduce,bst_weight=bst_weight)
            
            return {
                    "loss": loss,
                    "bootstrap_loss": bootstrap_loss.detach().mean(),
                    "flow_loss": loss.detach().mean(),
                    "t_flow_min": t_flow.min().item(),
                    "t_flow_max": t_flow.max().item(),
                    "t_flow_mean": t_flow.mean().item(),
                    "t_bst_min": t_bst.min().item(),
                    "t_bst_max": t_bst.max().item(),
                    "t_bst_mean": t_bst.mean().item(),
                    "r_min": r_bst.min().item(),
                    "r_max": r_bst.max().item(),
                    "r_mean": r_bst.mean().item(),
                    "s_min": s_bst.min().item(),
                    "s_max": s_bst.max().item(),
                    "s_mean": s_bst.mean().item(),
                    
                    "bootstrap_size": bst_size,
                    "progress": train_progress,
                }
            
        elif alg_type=="shortcut":
            assert annealing_mode!='none',"shortcut only support none annealing mode, pls use alg_type='shortcut_baseline'"
            assert bootstrap_every > 0, "bootstrap_every must be greater than 0 for shortcut"
            bst_size = annealing_bootstrap_every_fn(bs=bs, train_progress=train_progress, bootstrap_every=bootstrap_every, annealing_bootstrap_every=annealing_bootstrap_every)
            
            flow_size = bs - bst_size
            if verbose:
                print(f"bs: {bs}, bst_size: {bst_size}, flow_size: {flow_size}, train_progress: {train_progress}, bootstrap_every: {bootstrap_every}, annealing_bootstrap_every: {annealing_bootstrap_every}")
            
            xt_flow, t_flow, dt_flow, y_flow, ut_flow = _flow_branch(
                    x1=x1[:flow_size], x0=x0[:flow_size], y=None if y is None else y[:flow_size],  sc_kwargs=sc_kwargs, train_progress=train_progress, verbose=verbose)

            # bootstrap branch
            if use_ema_for_bootstrap:
                bootstrap_model = ema_model
            else:
                bootstrap_model = model

            xt_bst, t_bst, dt_bst, y_bst, ut_bst, x_t_prime_bst = _bootstrap_branch(
                x1[flow_size:], x0[flow_size:], y=None if y is None else y[flow_size:], bootstrap_model=bootstrap_model, train_progress=train_progress,sc_kwargs=sc_kwargs, verbose=verbose)
            if verbose:
                for i in range(bst_size):
                    print(f"t_bst[{i}]={t_bst[i]}, dt_bst[{i}]={dt_bst[i]}")

            if not dt_negative_loss:
                bst_size = len(dt_bst)
                xt = torch.cat([xt_flow, xt_bst], dim=0).detach()
                t = torch.cat([t_flow, t_bst], dim=0).detach()
                y = None if y is None else torch.cat([y_flow, y_bst], dim=0).detach()
                dt = torch.cat([dt_flow, dt_bst], dim=0).detach()
                ut = torch.cat([ut_flow, ut_bst], dim=0).detach()


                vt,zs_tilde = forward_wrapper(model,x=xt,t=t,dt=dt,y=y, **sc_kwargs)
                
                loss = (vt - ut)**2
                if False:
                    loss_weight_flow = torch.ones((flow_size,)).to(loss.device)
                    loss_weight_bst = weight_bootstrap_losses(dt=dt[flow_size:], weighting_mode=weighting_mode, progress=train_progress, num_steps=num_steps,verbose=verbose)
                    loss_weight = torch.cat([loss_weight_flow, loss_weight_bst], dim=0)
                    loss_weight = loss_weight.view(-1, *[1]*(loss.dim()-1))
                    loss = loss * loss_weight
            
                _loss_flow = loss[:flow_size]
                _loss_bootstrap = loss[flow_size:]
                _loss_bootstrap_neg = torch.tensor(0.0)
                loss = annealing_losses_fn(loss_flow=_loss_flow, loss_bootstrap=_loss_bootstrap,loss=loss, train_progress=train_progress, annealing_losses=annealing_losses, loss_reduce=loss_reduce,bst_weight=bst_weight)
            else:
                raise NotImplementedError("dt_negative_loss is not implemented yet")
                bst_size = len(dt_bst)
                xt = torch.cat([xt_flow, xt_bst,x_t_prime_bst], dim=0).detach()
                t = torch.cat([t_flow, t_bst,t_bst+0.5*dt_bst], dim=0).detach()
                y = None if y is None else torch.cat([y_flow, y_bst, y_bst], dim=0).detach()
                dt = torch.cat([dt_flow, dt_bst, -0.5*dt_bst], dim=0).detach()
                ut = torch.cat([ut_flow, ut_bst, -ut_bst], dim=0).detach()


                vt, zs_tilde = forward_wrapper(model,x=xt,t=t,dt=dt,y=y, **sc_kwargs)
                loss = (vt - ut)**2
        
                _loss_flow = loss[:flow_size]
                _loss_bootstrap = loss[flow_size:flow_size+bst_size]
                _loss_bootstrap_neg = loss[flow_size+bst_size:]
                loss = annealing_losses_fn(loss_flow=_loss_flow, loss_bootstrap=_loss_bootstrap,loss=loss, train_progress=train_progress, annealing_losses=annealing_losses, loss_reduce=loss_reduce,bst_weight=bst_weight)
                loss = loss + _loss_bootstrap_neg.mean()

            if sc_kwargs["use_repa"]:
                assert zs_tilde is not None, "zs_tilde must be not None for repa model"
                # projection loss
                zs = model_kwargs['zs']
                proj_loss = 0.0
                bsz = zs[0].shape[0]
                proj_coeff = sc_kwargs['repa_w']
                for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                    for j, (z_j, z_tilde_j) in enumerate(zip(z, z_tilde)):
                        z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1)
                        z_j = torch.nn.functional.normalize(z_j, dim=-1)
                        proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                proj_loss /= len(zs) * bsz
                loss = loss + proj_coeff * proj_loss
            else:
                proj_loss = torch.tensor(0.0,device=loss.device)
 
      

            loss_dict = {
                "loss": loss,
                "bootstrap_loss": _loss_bootstrap.detach().mean(),
                "bootstrap_loss_std": _loss_bootstrap.detach().std(),
                "flow_loss":      _loss_flow.detach().mean(),
                "flow_loss_std": _loss_flow.detach().std(),
                "bootstrap_neg_loss": _loss_bootstrap_neg.detach().mean(),
                "bootstrap_neg_loss_std": _loss_bootstrap_neg.detach().std(),
                "proj_loss": proj_loss.detach().mean(),
                "total_loss":     loss.detach().mean(),
                "dt_mean":   dt.to(torch.float32).mean().item(),
                "dt_max":         dt.max().item(),
                "dt_min":         dt.min().item(),
                "bst_weight":     bst_weight,
                "t_flow_mean":   t_flow.to(torch.float32).mean().item(),
                "t_flow_max":         t_flow.max().item(),
                "t_flow_min":         t_flow.min().item(),
                "t_bst_mean":   t_bst.to(torch.float32).mean().item(),
                "t_bst_max":         t_bst.max().item(),
                "t_bst_min":         t_bst.min().item(),
                "dt_bst_mean":   dt_bst.to(torch.float32).mean().item(),
                "dt_bst_max":         dt_bst.max().item(),
                "dt_bst_min":         dt_bst.min().item(),
                "dt_flow_mean":   dt_flow.to(torch.float32).mean().item(),
                "dt_flow_max":         dt_flow.max().item(),
                "dt_flow_min":         dt_flow.min().item(),
                "bootstrap_size": bst_size,
                "progress": train_progress,
            }
            return  loss_dict
        elif alg_type=='official':
            raise NotImplementedError("official is not implemented yet")
            assert bootstrap_every > 0, "bootstrap_every must be greater than 0 for official"
            y = sc_kwargs.pop('y')
            x_t, v_t, t, dt, y, bst_bs= create_targets_official(x1=x1,x0=x0, y=y, model=model, device=device, sc_kwargs=sc_kwargs, verbose=verbose)
            model.train()
            x_t = x_t.detach()
            v_t = v_t.detach()
            t = t.detach()
            dt = dt.detach()
            y = y.detach()

            if weighting_mode != "default":
                raise NotImplementedError("Weighting mode reverse not implemented")
            #for name, param in model.named_parameters():
            #    print(name, param.requires_grad)
            #dummy = torch.randn_like(x_t, requires_grad=False)
            #vt_est = model(x=dummy, t=t, dt=dt, y=y)
            vt_est = model(x=x_t, t=t, dt=dt, y=y)
            loss = (vt_est -  v_t)**2
            bootstrap_loss = loss[:bst_bs].mean()
            flow_loss = loss[bst_bs:].mean()
            loss = loss.mean()

            if False:
                print("x_t.requires_grad:", x_t.requires_grad)
                print("vt_est.requires_grad:", vt_est.requires_grad)
                print("vt_est.grad_fn:", vt_est.grad_fn)
                print("loss.requires_grad:", loss.requires_grad)
                print("loss.grad_fn:", loss.grad_fn)
            
            loss_dict = {
                "bootstrap_loss": bootstrap_loss.detach().mean(),
                "flow_loss": flow_loss.detach().mean(),
                "total_loss": loss.detach().mean(),
            }
            return loss, loss_dict
        else:
            raise NotImplementedError(f"alg_type {alg_type} not implemented")


@torch.no_grad()
def _bootstrap_branch(x1: Tensor, x0: Tensor, y: Tensor,train_progress, sc_kwargs, bootstrap_model: Optional[nn.Module] = None,  verbose: bool = False):
        if sc_kwargs['bootstrap_branch'] == 'default':
            return _bootstrap_branch_default(x1, x0, y, train_progress, sc_kwargs, bootstrap_model, verbose)
        elif sc_kwargs['bootstrap_branch'] == 'immboost':
            return _bootstrap_branch_immboost(x1, x0, y, train_progress, sc_kwargs, bootstrap_model, verbose)
        else:
            raise NotImplementedError(f"bootstrap_branch {sc_kwargs['bootstrap_branch']} not implemented")




def _bootstrap_branch_immboost(x1: Tensor, x0: Tensor, y: Tensor,train_progress, sc_kwargs, bootstrap_model: Optional[nn.Module] = None,  verbose: bool = False):
        
        use_ema_for_bootstrap = sc_kwargs['use_ema_for_bootstrap']
        weighting_mode = sc_kwargs['weighting_mode']
        annealing_mode = sc_kwargs['annealing_mode']
        num_steps = sc_kwargs['num_steps']
        dt_negative = sc_kwargs['dt_negative']

        bs, device, dtype = len(x1), x1.device, x1.dtype
        t, twodt, twodt_base,  onedt, onedt_base = sample_delta(batch_size=bs, time_steps=num_steps, device=device, annealing_mode=annealing_mode, train_progress=train_progress, dt_negative=dt_negative,verbose=verbose)
        
        t_vec = t.view(-1, *[1]*(x0.ndim-1))
        dt_vec = onedt.view(-1, *[1]*(x0.ndim-1))
        
        if verbose:
            print("x0.shape", x0.shape, "x1.shape", x1.shape,"t.shape", t.shape,"dt.shape", onedt.shape,"y.shape", y.shape)
        x_t = (1 - (1-eps) * t_vec)*x0 + t_vec*x1
        t_prime_vec = t_vec + dt_vec
        t_prime = t + onedt
        x_t_prime = (1 - (1-eps) * t_prime_vec)*x0 + t_prime_vec*x1 
        
        v_teacher = bootstrap_model(x=x_t, t=t, dt=onedt, y=y, **sc_kwargs)
        v_teacher = v_teacher.detach()
        return x_t_prime, t_prime, onedt, y, v_teacher




def _bootstrap_branch_default(x1: Tensor, x0: Tensor, y: Tensor,train_progress, sc_kwargs, bootstrap_model: Optional[nn.Module] = None,  verbose: bool = False, clamp_value: float = 4.0):
        """
        Bootstrap branch of the model. This function implements a two-step prediction process
        that generates target vectors for training using the model's own predictions.

        Args:
            x0: shape (bs, *dim), represents the source minibatch (noise)
            x1: shape (bs, *dim), represents the target minibatch (data)
            sc_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)

        Returns:
            xt: shape (bs, *dim), sampled point along the path
            t: shape (bs,), sampled time points
            ut: shape (bs, *dim), target vector field (average of two predictions)
            vt: shape (bs, *dim), model prediction
            dt: shape (bs,), time step sizes
            dt_base: shape (bs,), base time step sizes
            dt_bootstrap: shape (bs,), bootstrap time step sizes
            loss_weight: shape (bs,), weights for the bootstrap loss
            ut_negative: shape (bs, *dim) or None, negative time step predictions if dt_negative=True
        
        Process:
        1. Samples time steps and computes step sizes based on annealing mode
        2. Generates first prediction v_b1 at time t
        3. Takes a step to t2 = t + dt_bootstrap using v_b1
        4. Generates second prediction v_b2 at time t2
        5. Averages v_b1 and v_b2 to create target ut
        6. Optionally computes negative time step predictions
        7. Generates final prediction vt for loss computation
        """
        use_ema_for_bootstrap = sc_kwargs['use_ema_for_bootstrap']
        weighting_mode = sc_kwargs['weighting_mode']
        annealing_mode = sc_kwargs['annealing_mode']
        num_steps = sc_kwargs['num_steps']
        dt_negative = sc_kwargs['dt_negative']
        etazero=sc_kwargs["etazero"]   

        
        bs, device, dtype = len(x1), x1.device, x1.dtype
        t, twodt, twodt_base,  onedt, onedt_base = sample_delta(batch_size=bs, time_steps=num_steps, device=device, etazero=etazero,annealing_mode=annealing_mode,  train_progress=train_progress, dt_negative=dt_negative,verbose=verbose)

        #########################################################
        if verbose:
            for i in range(bs):
                print(f"bootstrap: t[i]: {t[i]}, twodt[i]: {twodt[i]}, twodt_base[i]: {twodt_base[i]}, onedt[i]: {onedt[i]}, onedt_base[i]: {onedt_base[i]}")


        xt = compute_xt_fn(x0=x0, x1=x1, t=t)
        v_b1, _ = forward_wrapper(bootstrap_model,x=xt,t=t,dt=onedt.float(),y=y, **sc_kwargs)
       
        t2 = t + onedt
        xt2 = xt + onedt.view(-1, 1, 1, 1) * v_b1
        xt2 = xt2.clamp(-clamp_value, clamp_value)
        
        
        
        v_b2,_ = forward_wrapper(bootstrap_model, x=xt2, t=t2, dt=onedt.float(), y=y, **sc_kwargs)
       
        ut = (v_b1 + v_b2) / 2    
        ut = ut.clamp(-clamp_value, clamp_value).detach()
        #x_t_prime = xt2 + onedt.view(-1, 1, 1, 1) * v_b2
        #x_t_prime = x_t_prime.clamp(-clamp_value, clamp_value)
        

        return xt, t, twodt, y, ut,xt2





def _bootstrap_branch_eta_space(x1: Tensor, x0: Tensor, y: Tensor,train_progress, sc_kwargs, bootstrap_model: Optional[nn.Module] = None,  verbose: bool = False, clamp_value: float = 4.0):
        """
        Bootstrap branch of the model. This function implements a two-step prediction process
        that generates target vectors for training using the model's own predictions.

        Args:
            x0: shape (bs, *dim), represents the source minibatch (noise)
            x1: shape (bs, *dim), represents the target minibatch (data)
            sc_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)

        Returns:
            xt: shape (bs, *dim), sampled point along the path
            t: shape (bs,), sampled time points
            ut: shape (bs, *dim), target vector field (average of two predictions)
            vt: shape (bs, *dim), model prediction
            dt: shape (bs,), time step sizes
            dt_base: shape (bs,), base time step sizes
            dt_bootstrap: shape (bs,), bootstrap time step sizes
            loss_weight: shape (bs,), weights for the bootstrap loss
            ut_negative: shape (bs, *dim) or None, negative time step predictions if dt_negative=True
        
        Process:
        1. Samples time steps and computes step sizes based on annealing mode
        2. Generates first prediction v_b1 at time t
        3. Takes a step to t2 = t + dt_bootstrap using v_b1
        4. Generates second prediction v_b2 at time t2
        5. Averages v_b1 and v_b2 to create target ut
        6. Optionally computes negative time step predictions
        7. Generates final prediction vt for loss computation
        """
        use_ema_for_bootstrap = sc_kwargs['use_ema_for_bootstrap']
        weighting_mode = sc_kwargs['weighting_mode']
        annealing_mode = sc_kwargs['annealing_mode']
        num_steps = sc_kwargs['num_steps']
        dt_negative = sc_kwargs['dt_negative']
        etazero=sc_kwargs["etazero"]   

        
        bs, device, dtype = len(x1), x1.device, x1.dtype
        t, dt1, dt2 = sample_delta_eta(batch_size=bs, time_steps=num_steps, device=device, etazero=etazero,annealing_mode=annealing_mode,  train_progress=train_progress, dt_negative=dt_negative,verbose=verbose)

        #########################################################
        if verbose:
            for i in range(bs):
                print(f"bootstrap: t[i]: {t[i]}, dt1[i]: {dt1[i]}, dt2[i]: {dt2[i]}")


        xt = compute_xt_fn(x0=x0, x1=x1, t=t)
        v_b1, _ = forward_wrapper(bootstrap_model,x=xt,t=t,dt=dt1.float(),y=y, **sc_kwargs)
       
        t2 = t + dt1
        xt2 = xt + dt1.view(-1, 1, 1, 1) * v_b1
        xt2 = xt2.clamp(-clamp_value, clamp_value)
        
        
        
        v_b2,_ = forward_wrapper(bootstrap_model, x=xt2, t=t2, dt=dt2.float(), y=y, **sc_kwargs)
       
        ut = (v_b1 + v_b2) / 2    
        ut = ut.clamp(-clamp_value, clamp_value).detach()
        #x_t_prime = xt2 + onedt.view(-1, 1, 1, 1) * v_b2
        #x_t_prime = x_t_prime.clamp(-clamp_value, clamp_value)
        

        return xt, t, dt1+dt2, y, ut,xt2


def _flow_branch_shortcut_baseline(x1: Tensor, x0: Tensor, y: Tensor, sc_kwargs, train_progress: float, verbose=False):
        """
        For each sample in the batch, this branch performs the following steps:
        1. Computes a constant dt value (dt_flow_val) as log₂(num_steps). For example, if num_steps is 128, dt_flow_val will be 7.
        2. Samples a discrete timestep t for each example by drawing a random integer in the range [0, dt_flow_val)
            and normalizing it by dividing by dt_flow_val. This yields t values on [0, 1] in increments of 1/dt_flow_val.
        3. Computes an interpolated sample xt from x0 (source, typically noise) and x1 (target data) using the schedule:
                xt = alpha(t) * x1 + sigma(t) * x0,
            where alpha(t)=t and sigma(t)=1-t (for a linear schedule).
        4. Computes the corresponding target conditional vector field ut based on x0, x1, and the sampled t.
        5. Performs a forward pass through the model using xt, t, and a fixed dt equal to dt_flow_val for all samples,
            yielding the predicted vector field vt.

        Args:
            x0: shape (bs, *dim), represents the source minibatch (noise)
            x1: shape (bs, *dim), represents the target minibatch (data)
            sc_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)
        Returns:
            v_target: shape (bs, *dim), represents the target vector field
            xt: shape (bs, *dim), represents the sampled point along the time-dependent density p_t
        """
        num_steps=sc_kwargs["num_steps"]
        bs, device = x1.shape[0], x1.device
        _dt = 1.0 * torch.ones(bs, device=device, dtype=torch.int32)/num_steps #use minimum dt for flow
        t = torch.randint(0, num_steps, (bs,), device=device, dtype=torch.int32)
        t = t.float() / num_steps
        xt = compute_xt_fn(x0=x0, x1=x1, t=t)
        ut = compute_ut_fn(x0=x0, x1=x1, t=t)
            
            

        return xt, t, _dt, y, ut




def _flow_branch(x1: Tensor, x0: Tensor, y: Tensor, sc_kwargs, train_progress: float, verbose=False):
        """
        For each sample in the batch, this branch performs the following steps:
        1. Computes a constant dt value (dt_flow_val) as log₂(num_steps). For example, if num_steps is 128, dt_flow_val will be 7.
        2. Samples a discrete timestep t for each example by drawing a random integer in the range [0, dt_flow_val)
            and normalizing it by dividing by dt_flow_val. This yields t values on [0, 1] in increments of 1/dt_flow_val.
        3. Computes an interpolated sample xt from x0 (source, typically noise) and x1 (target data) using the schedule:
                xt = alpha(t) * x1 + sigma(t) * x0,
            where alpha(t)=t and sigma(t)=1-t (for a linear schedule).
        4. Computes the corresponding target conditional vector field ut based on x0, x1, and the sampled t.
        5. Performs a forward pass through the model using xt, t, and a fixed dt equal to dt_flow_val for all samples,
            yielding the predicted vector field vt.

        Args:
            x0: shape (bs, *dim), represents the source minibatch (noise)
            x1: shape (bs, *dim), represents the target minibatch (data)
            sc_kwargs: additional arguments for the conditional flow
                network (e.g. conditioning information)
        Returns:
            v_target: shape (bs, *dim), represents the target vector field
            xt: shape (bs, *dim), represents the sampled point along the time-dependent density p_t
        """
        flowloss_pyramid=sc_kwargs["flowloss_pyramid"]
        dt_negative=sc_kwargs["dt_negative"]
        num_steps=sc_kwargs["num_steps"]
        annealing_mode=sc_kwargs["annealing_mode"]
        etazero=sc_kwargs["etazero"]
        flowloss_ct=sc_kwargs["flowloss_ct"]
        bs, device = len(x1), x1.device

        if flowloss_pyramid:
            #_dt = 1.0 * torch.ones(bs, device=dev, dtype=torch.int32)/num_steps #use minimum dt for flow branch to query the vector
            #t = torch.randint(0, num_steps, (bs,), device=dev, dtype=torch.int32)
            #t = t.float() / num_steps
            t, twodt, twodt_base,  onedt, onedt_base = sample_delta(batch_size=bs, time_steps=num_steps, device=device, annealing_mode=annealing_mode, etazero=etazero,train_progress=train_progress, dt_negative=dt_negative, verbose=verbose)

            xt = compute_xt_fn(x0=x0, x1=x1, t=t)
            ut = compute_ut_fn(x0=x0, x1=x1, t=t)
            if False:
                mask = torch.randint(0, 2, (bs,), device=onedt.device, dtype=torch.bool)
                _dt = torch.where(mask, onedt, twodt)
            else:
                _dt = twodt
            if verbose:
                for i  in range(bs):    
                    print(f"flowloss_pyramid: t: {t[i]}, _dt: {_dt[i]}, twodt: {twodt[i]}, onedt: {onedt[i]}")
            
        else:
            if flowloss_ct:
                t = torch.rand(bs, device=device, dtype=torch.float32)
            else:
                t = torch.randint(0, num_steps, (bs,), device=device, dtype=torch.int32)
                t = t.float() / num_steps
            if not etazero:
                raise NotImplementedError("etazero should be False here")
                _dt = 1.0 * torch.ones(bs, device=device, dtype=torch.int32)/num_steps #use minimum dt for flow
            else:
                _dt =torch.zeros(bs, device=device, dtype=torch.int32)
            xt = compute_xt_fn(x0=x0, x1=x1, t=t)
            ut = compute_ut_fn(x0=x0, x1=x1, t=t)
            
            

        return xt, t, _dt, y, ut


def sample_delta(batch_size: int, time_steps: int,  device: torch.device, etazero: bool=None ,annealing_mode: str = "none",  train_progress: float = 0.0, dt_negative: bool = False, verbose: bool = False):
        """
        Sample the time step sizes for the diffusion process.

        Args:
            batch_size: The batch size.
            time_steps: The number of diffusion time steps.
            device: The device to run the operation on.
        Returns:
            t: The time step for each example.
            dt: 1 / 2^dt_base for each example.
            dt_base: The base exponent for each example.
            twodt: Half the dt for bootstrap computations.
            twodt_base: dt_base + 1 for bootstrap computations.
        """
        L = int(np.log2(time_steps))
        idx = torch.arange(L, dtype=torch.float32, device=device)
        assert etazero is not None, "etazero must be provided"
        def _shortcut_sample():
            dt_base_vals = torch.arange(L, dtype=torch.int32, device=device)
            twodt_base = (L - 1) - dt_base_vals
            repeat_factor = batch_size // L
            twodt_base = twodt_base.repeat(repeat_factor)#repeat 
            leftover = batch_size - twodt_base.shape[0]
            if leftover > 0:
                twodt_base = torch.cat([twodt_base, torch.zeros(leftover, device=device, dtype=torch.int32)], dim=0)

            twodt = 1.0 / (2 ** twodt_base.float())
            onedt_base = twodt_base + 1
            onedt = twodt / 2.0

            dt_sections = 2 ** twodt_base.abs().float()# to make that the dt can be negative
            t = torch.floor(torch.rand(batch_size, device=device) * dt_sections).to(torch.int32)
            t = t.float() / dt_sections
            return t, twodt, twodt_base,  onedt, onedt_base
        
        if annealing_mode == "none":
            t, twodt, twodt_base,  onedt, onedt_base = _shortcut_sample()
        elif annealing_mode in ["continuous"] or annealing_mode.startswith("continuous_"):
            eps = 1e-5
            if annealing_mode == "continuous":
                _real_progress = train_progress
            elif annealing_mode.startswith("continuous_"):
                _ratio = float(annealing_mode.split("_")[1])
                _real_progress = min(train_progress/_ratio, 1.0)
            else:
                raise ValueError(f"Invalid annealing mode: {annealing_mode}")
            max_dt_min = 1.0/time_steps
            max_dt_max = 0.5-4*eps
            max_dt =max_dt_min + (max_dt_max - max_dt_min) * _real_progress
            dt = torch.rand(batch_size, device=device, dtype=torch.float32) * (max_dt - max_dt_min) + max_dt_min
            t_min =eps
            t_max = 1-dt*2
            t = torch.rand(batch_size, device=device, dtype=torch.float32) *(t_max - t_min) + t_min
            twodt = dt*2
            onedt = dt
            onedt_base = None
            twodt_base = None
            #assert dt_negative == False, "dt_negative is not supported for continuous annealing"
            if verbose:
                for i in range(min(batch_size,16)):
                    print(f"progress: {train_progress:0.2f}, max_dt: {max_dt:.3f},  dt[i]: {dt[i]:.3f}, twodt[i]: {twodt[i]:.3f}, t[i]: {t[i]:.3f}, t+2dt[i]: {t[i]+2*dt[i]:.3f}")
        elif annealing_mode.startswith("beta"):
            beta_value = float(annealing_mode.split("_")[1])
            twodt = sample_beta_batch(batch_size, alpha=beta_value, beta=beta_value,device=device)
            onedt = twodt / 2.0

            t = torch.rand(batch_size, device=device)*(1-twodt)
            onedt_base = None
            twodt_base = None

        elif annealing_mode == "delta_max":
            progress_index = int(train_progress*L)+1
            dt_base_vals = torch.arange(L, dtype=torch.int32, device=device)
            twodt_base = (L - 1) - dt_base_vals
            twodt_base = twodt_base[:progress_index]
            if verbose:
                print("twodt_base filtered: ", twodt_base)
            #######
            repeat_factor = batch_size // len(twodt_base)
            twodt_base = twodt_base.repeat(repeat_factor+1)#repeat 
            twodt_base = twodt_base[:batch_size]

            twodt = 1.0 / (2 ** twodt_base.float())
            onedt_base = twodt_base + 1
            onedt = twodt / 2.0

            dt_sections = 2 ** twodt_base.abs().float()# to make that the dt can be negative
            t = torch.floor(torch.rand(batch_size, device=device) * dt_sections).to(torch.int32)
            t = t.float() / dt_sections
            if verbose:
                for i in range(min(batch_size,64)):
                    print(f"progress: {train_progress:0.2f}, dt[i]: {onedt[i]:.3f}, t[i]: {t[i]:.3f}")
                print("max dt: ", max(onedt), "min dt: ", min(onedt),'mean dt: ', onedt.mean().item())

        elif annealing_mode in ["dcontinuous"] or annealing_mode.startswith("dcontinuous_"):
            eps = 1e-5
            nfe1_ratio = 1/math.log2(time_steps)
            nfe1_bs = int(batch_size*nfe1_ratio)
            assert nfe1_bs > 0, "nfe1_bs must be greater than 0"
            if annealing_mode == "dcontinuous":
                _real_progress = train_progress
            elif annealing_mode.startswith("dcontinuous_"):
                _ratio = float(annealing_mode.split("_")[1])
                _real_progress = min(train_progress/_ratio, 1.0)
            else:
                raise ValueError(f"Invalid annealing mode: {annealing_mode}")
            max_dt_min = 0 if etazero else 1.0/time_steps
            max_dt_max = 0.5-4*eps
            max_dt =max_dt_min + (max_dt_max - max_dt_min) * _real_progress
            
            dt = torch.rand(batch_size, device=device, dtype=torch.float32) * (max_dt - max_dt_min) + max_dt_min
            if max_dt == max_dt_max:
                dt[:nfe1_bs] = max_dt_max  
            t_min =eps
            t_max = 1-dt*2
            t = torch.rand(batch_size, device=device, dtype=torch.float32) *(t_max - t_min) + t_min
            twodt = dt*2
            onedt = dt
            onedt_base = None
            twodt_base = None
            #assert dt_negative == False, "dt_negative is not supported for continuous annealing"
            if verbose:
                for i in range(min(batch_size,16)):
                    print(f"progress: {train_progress:0.2f}, max_dt: {max_dt:.3f},  dt[i]: {dt[i]:.3f}, twodt[i]: {twodt[i]:.3f}, t[i]: {t[i]:.3f}, t+2dt[i]: {t[i]+2*dt[i]:.3f}")
        elif annealing_mode in ["sc_after_continuous"] or annealing_mode.startswith("sc_after_continuous_"):
            eps = 1e-5
            nfe1_ratio = 1/math.log2(time_steps)
            nfe1_bs = int(batch_size*nfe1_ratio)
            assert nfe1_bs > 0, "nfe1_bs must be greater than 0"
            if annealing_mode == "sc_after_continuous":
                _real_progress = train_progress
            elif annealing_mode.startswith("sc_after_continuous_"):
                _ratio = float(annealing_mode.split("_")[-1])
                _real_progress = min(train_progress/_ratio, 1.0)
            else:
                raise ValueError(f"Invalid annealing mode: {annealing_mode}")
            max_dt_min = 0 if etazero else 1.0/time_steps
            max_dt_max = 0.5-4*eps
            max_dt =max_dt_min + (max_dt_max - max_dt_min) * _real_progress
            
            dt = torch.rand(batch_size, device=device, dtype=torch.float32) * (max_dt - max_dt_min) + max_dt_min
            if max_dt == max_dt_max:
                t, twodt, twodt_base,  onedt, onedt_base = _shortcut_sample()
            else:
                t_min =eps
                t_max = 1-dt*2
                t = torch.rand(batch_size, device=device, dtype=torch.float32) *(t_max - t_min) + t_min
                twodt = dt*2
                onedt = dt
                onedt_base = None
                twodt_base = None
                #assert dt_negative == False, "dt_negative is not supported for continuous annealing"
            if verbose:
                for i in range(min(batch_size,16)):
                    print(f"progress: {train_progress:0.2f}, max_dt: {max_dt:.3f},  dt[i]: {dt[i]:.3f}, twodt[i]: {twodt[i]:.3f}, t[i]: {t[i]:.3f}, t+2dt[i]: {t[i]+2*dt[i]:.3f}")
        
        elif annealing_mode in ["continuousl2s"] or annealing_mode.startswith("continuousl2s_"):
            eps = 1e-5
            if annealing_mode == "continuousl2s":
                _real_progress = train_progress
            elif annealing_mode.startswith("continuousl2s_"):
                _ratio = float(annealing_mode.split("_")[1])
                _real_progress = min(train_progress/_ratio, 1.0)
            else:
                raise ValueError(f"Invalid annealing mode: {annealing_mode}")
            max_dt_min = 1.0/time_steps
            max_dt_max = 0.5-4*eps
            min_dt =max_dt_max - (max_dt_max - max_dt_min) * _real_progress
            dt = torch.rand(batch_size, device=device, dtype=torch.float32) * (max_dt_max - min_dt) + min_dt
            t_min =eps
            t_max = 1-dt*2
            t = torch.rand(batch_size, device=device, dtype=torch.float32) *(t_max - t_min) + t_min
            twodt = dt*2
            onedt = dt
            onedt_base = None
            twodt_base = None
            #assert dt_negative == False, "dt_negative is not supported for continuous annealing"
            if verbose:
                for i in range(min(batch_size,16)):
                    print(f"progress: {train_progress:0.2f}, min_dt: {min_dt:.3f},  dt[i]: {dt[i]:.3f}, twodt[i]: {twodt[i]:.3f}, t[i]: {t[i]:.3f}, t+2dt[i]: {t[i]+2*dt[i]:.3f}")
    

        elif annealing_mode in ["uniform", "reverse","reverse_late","reverse_mid","uniform_late"]:
            _progress = torch.tensor(train_progress, dtype=torch.float32, device=device)
            if annealing_mode == "uniform":
                s0, k = 0.2, 0.05
                alpha = torch.sigmoid((_progress - s0) / k)
                logits = (1.0 - alpha) * (-idx) 
            elif annealing_mode == "uniform_late":
                if _progress < 0.7:
                    s0, k = 0.7, 0.3
                    alpha = torch.sigmoid((_progress - s0) / k)
                    logits = (1.0 - alpha) * (-idx) + alpha * (+idx)
                else:
                    logits = torch.zeros_like(idx)  # Uniform distribution
            elif annealing_mode == "reverse":
                s0, k = 0.3, 0.3
                alpha = torch.sigmoid((_progress - s0) / k)
                logits = (1.0 - alpha) * (-idx) + alpha * (+idx)
            elif annealing_mode == "reverse_mid":
                s0, k = 0.5, 0.3
                alpha = torch.sigmoid((_progress - s0) / k)
                logits = (1.0 - alpha) * (-idx) + alpha * (+idx)
            elif annealing_mode == "reverse_late":
                s0, k = 0.7, 0.3
                alpha = torch.sigmoid((_progress - s0) / k)
                logits = (1.0 - alpha) * (-idx) + alpha * (+idx)
            
            else:
                raise ValueError(f"Invalid annealing mode: {annealing_mode}")
            
            probs = torch.softmax(logits, dim=0)
            sampled_indices = torch.multinomial(probs, batch_size, replacement=True)
            twodt_base = (L - 1) - sampled_indices
            
            twodt = 1.0 / (2 ** twodt_base.float())
            onedt_base = twodt_base + 1
            onedt = twodt / 2.0
            assert twodt.min() > 0 and twodt.max() <= 1
            assert onedt.min() > 0 and onedt.max() <= 1

            dt_sections = 2 ** twodt_base.abs().float()# to make that the dt can be negative
            t = torch.floor(torch.rand(batch_size, device=device) * dt_sections).to(torch.int32)
            t = t.float() / dt_sections
            
        else:
            raise ValueError(f"Invalid annealing mode: {annealing_mode}")
        
        
        if verbose:
            for i in range(min(batch_size,16)):
                print(f"progress: {train_progress:0.2f}, index: {i}, bs: {batch_size}, annealing_mode: {annealing_mode}, L: {L}, t[i]: {t[i]},  onedt[i]/twodt[i]: {onedt[i]}/{twodt[i]}")
                if twodt_base is not None:
                    print(f"twodt_base[i]: {twodt_base[i]}, twodt[i]: {twodt[i]}")

        assert onedt.min() >= 0 and onedt.max() <= 1
        assert twodt.min() >= 0 and twodt.max() <= 1
        if dt_negative:
            t_pos = t 
            t_neg = 1-t

            neg_sign = 2 * (torch.rand(batch_size, device=device) < 0.5).float() - 1
            onedt = onedt * neg_sign
            if onedt_base is not None:
                onedt_base = onedt_base * neg_sign
            if twodt is not None:
                twodt = twodt * neg_sign
            if twodt_base is not None:
                twodt_base = twodt_base * neg_sign
            t = torch.where(neg_sign==1.0, t_pos, t_neg)
            if verbose:
                for i in range(batch_size):
                    print(f"dt_negative: t[i]: {t[i]}, onedt[i]: {onedt[i]}, neg_sign[i]: {neg_sign[i]}")
        else:
            assert onedt.min() >= 0 and onedt.max() <= 1, f"onedt.min(): {onedt.min()}, onedt.max(): {onedt.max()}"
        assert t.min() >= 0 and t.max() <= 1
        
        return t, twodt, twodt_base,  onedt, onedt_base


def get_eta_mapping_fn(name='fm',eps=1e-6, t_eps=1e-3, batch_size=None, device=None, progress=None, verbose=False):
    def t_to_eta(t):#compute under the condition of t=0 is data, t=1 is noise
            # Flow Matching, x_t = (1-t)*x_0 + t*x_1
            assert t.min() >= 0 and t.max() <= 1, f"t.min(): {t.min()}, t.max(): {t.max()}"
            _noise = 1 - t
            _data = t
            eta_t = _data / (_noise+eps)
            return eta_t
    
    def eta_to_t(eta):
            assert eta.min() >= eta_min and eta.max() <= eta_max, f"eta.min(): {eta.min()}, eta.max(): {eta.max()}"
            # Flow Matching, x_t = (1-t)*x_0 + t*x_1
            #eta = sigma_t / alpha_t,
            t = eta/(1+eta)
            assert t.min() >= 0 and t.max() <= 1, f"t.min(): {t.min()}, t.max(): {t.max()}"
            return t
    
    if False:
        eta_min = t_to_eta(torch.tensor(0.0))
        eta_max = t_to_eta(torch.tensor(1.0-t_eps))
    elif False:
        eta_min = t_to_eta(torch.tensor(0.0))
        eta_max = 160/2**12 #t_to_eta(torch.tensor(1.0-t_eps))
    elif True:
        eta_max = t_to_eta(torch.tensor(1.0-t_eps))
        eta_min = eta_max - 500 #t_to_eta(torch.tensor(0.0))

    eta_range = eta_max - eta_min
    assert eta_range > 0, f"eta_range: {eta_range}"
    if verbose:
        print(f"eta_min: {eta_min}, eta_max: {eta_max}")
    max_eta_dt2 = eta_range*progress
    
   
    _current_eta_dt2 = torch.rand(batch_size, device=device) * max_eta_dt2
    eta_t = torch.rand(batch_size, device=device) * (eta_range - _current_eta_dt2) + eta_min
    _t = eta_to_t(eta_t)
    _dt1 = eta_to_t(eta_t + _current_eta_dt2*0.5) - _t
    _dt2 = eta_to_t(eta_t + _current_eta_dt2) - _t
    if verbose:
        for i in range(batch_size):
            print("eta_max",eta_max,"eta_range",eta_range.item(), "eta_t[i]",eta_t[i].item(),"_current_eta_dt2",_current_eta_dt2[i].item())
            print(f"progress: {progress:.2f}, eta_t[i]: {eta_t[i]:.3f}, _t[i]: {_t[i]:.3f}, _dt1[i]: {_dt1[i]:.3f}, _dt2[i]: {_dt2[i]:.3f}")
            print("*"*100)
    return _t, _dt1, _dt2
        
    
    
    

    
    

def sample_delta_eta(batch_size: int,  device: torch.device, annealing_mode: str = "none",  train_progress: float = 0.0,  verbose: bool = False, name: str = 'fm'):
        t,dt1,dt2 = get_eta_mapping_fn(name=name,verbose=verbose,progress=train_progress,batch_size=batch_size,device=device)
        return t, dt2, None, dt1, None 



def sample_immfm_dmdn(batch_size: int, device: torch.device, annealing_mode: str = "continuous",  train_progress: float = 0.0, dt_negative: bool = False, num_steps: int = None, verbose: bool = False):
        """
        Sample the time step sizes for the diffusion process.

        Args:
            batch_size: The batch size.
            time_steps: The number of diffusion time steps.
            device: The device to run the operation on.
        Returns:
            t: The time step for each example.
            dt: 1 / 2^dt_base for each example.
            dt_base: The base exponent for each example.
            twodt: Half the dt for bootstrap computations.
            twodt_base: dt_base + 1 for bootstrap computations.
        """
       
    
        if annealing_mode in ["continuous"] or annealing_mode.startswith("continuous_"):
            eps = 1e-5
            if annealing_mode == "continuous":
                _real_progress = train_progress
            elif annealing_mode.startswith("continuous_"):
                _progress_threshold  = float(annealing_mode.split("_")[1])
                _real_progress = min(train_progress/_progress_threshold, 1.0)

            else:
                raise ValueError(f"Invalid annealing mode: {annealing_mode}")
            
            dn_scale = 1.0/num_steps
            dn = dn_scale*torch.ones(batch_size, device=device, dtype=torch.float32)
            dm_max = max(_real_progress, eps)
            dm_max = min(dm_max, 1-dn_scale)
            dm = torch.rand(batch_size, device=device, dtype=torch.float32) * (dm_max - eps) + eps
            t = torch.rand(batch_size, device=device, dtype=torch.float32) * (1-dm-dn_scale)
            if verbose:
                for i in range(batch_size):
                    print(f"progress: {train_progress:0.2f}, t[i]: {t[i]:.3f}, dm[i]/dm_max: {dm[i]:.3f}/{dm_max:.3f}, dn[i]: {dn[i]:.3f}")
            assert t.min() >= 0 and t.max() <= 1, f"t.min(): {t.min()}, t.max(): {t.max()}"
            assert (t+dm+dn).min() >= 0 and (t+dm+dn).max() <= 1, f"(t+dm+dn).min(): {(t+dm+dn).min()}, (t+dm+dn).max(): {(t+dm+dn).max()}"
        elif annealing_mode in ["imm"] and False:
            eps = 1e-5
            _t = torch.rand(batch_size, device=device, dtype=torch.float32)*(1-eps)+eps 
            _s = torch.rand(batch_size, device=device, dtype=torch.float32)*_t*(1-eps)+eps
            def compute_r(_t, _s, eta_max=160, eta_min=0, k=11):#compute under the condition of t=0 is data, t=1 is noise
                alpha_t, sigma_t = 1-_t, _t 

                eta_t = sigma_t / alpha_t
                eps = (eta_max - eta_min) / 2**k
                eta = eta_t - eps

            
                _r = eta / (1.0 + eta)

                return torch.maximum(_s, _r)
            _r = compute_r(_t, _s)
            for _i  in range(batch_size):
                print("t,r,s",_t[_i],_r[_i],_s[_i])
            t = 1-_s 
            dm = _r - _s
            dn = _t - _r 
            for i in range(batch_size):
                print("t,dm,dn",t[i],dm[i],dn[i])
            exit()

            if verbose:
                for i in range(batch_size):
                    print(f"progress: {train_progress:0.2f}, t[i]: {t[i]:.3f}, dm[i]/dm_max: {dm[i]:.3f}/{dm_max:.3f}, dn[i]: {dn[i]:.3f}")
            assert t.min() >= 0 and t.max() <= 1, f"t.min(): {t.min()}, t.max(): {t.max()}"
            assert (t+dm+dn).min() >= 0 and (t+dm+dn).max() <= 1, f"(t+dm+dn).min(): {(t+dm+dn).min()}, (t+dm+dn).max(): {(t+dm+dn).max()}"
        elif annealing_mode in ["continuousuni"]:
            eps = 1e-5
            if annealing_mode == "continuousuni":
                _real_progress = train_progress
            pass 
            dn_scale = 1.0/num_steps
            dn = dn_scale*torch.ones(batch_size, device=device, dtype=torch.float32)
            dm_min = dn_scale
            dm_max = 1-dn_scale
            dm = torch.rand(batch_size, device=device, dtype=torch.float32) * (dm_max - dm_min) + dm_min
            t = torch.rand(batch_size, device=device, dtype=torch.float32) * (1-dm-dn_scale)
            if verbose:
                for i in range(batch_size):
                    print(f"progress: {train_progress:0.2f}, t[i]: {t[i]:.3f}, dm[i]/dm_max/dm_min: {dm[i]:.3f}/{dm_max:.3f}/{dm_min:.3f}, dn[i]: {dn[i]:.3f}")
            assert t.min() >= 0 and t.max() <= 1, f"t.min(): {t.min()}, t.max(): {t.max()}"
            assert (t+dm+dn).min() >= 0 and (t+dm+dn).max() <= 1, f"(t+dm+dn).min(): {(t+dm+dn).min()}, (t+dm+dn).max(): {(t+dm+dn).max()}"

            

        elif annealing_mode in ["continuousl2s"] or annealing_mode.startswith("continuousl2s_"):
            eps = 1e-5
            if annealing_mode == "continuousl2s":
                _real_progress = train_progress
            elif annealing_mode.startswith("continuousl2s_"):
                _progress_threshold  = float(annealing_mode.split("_")[1])
                _real_progress = min(train_progress/_progress_threshold, 1.0)

            else:
                raise ValueError(f"Invalid annealing mode: {annealing_mode}")
            
            dn_scale = 1.0/128
            dn = dn_scale*torch.ones(batch_size, device=device, dtype=torch.float32)
            dm_min = max(1-_real_progress, eps)-dn_scale
            dm_max = 1-dn_scale
            dm = torch.rand(batch_size, device=device, dtype=torch.float32) * (dm_max - dm_min) + dm_min
            t = torch.rand(batch_size, device=device, dtype=torch.float32) * (1-dm-dn_scale)
            if verbose:
                for i in range(batch_size):
                    print(f"progress: {train_progress:0.2f}, t[i]: {t[i]:.3f}, dm[i]/dm_max/dm_min: {dm[i]:.3f}/{dm_max:.3f}/{dm_min:.3f}, dn[i]: {dn[i]:.3f}")
            assert t.min() >= 0 and t.max() <= 1, f"t.min(): {t.min()}, t.max(): {t.max()}"
            assert (t+dm+dn).min() >= 0 and (t+dm+dn).max() <= 1, f"(t+dm+dn).min(): {(t+dm+dn).min()}, (t+dm+dn).max(): {(t+dm+dn).max()}"
        else:
            raise ValueError(f"Invalid annealing mode: {annealing_mode}")
        
        if dt_negative:
            t_neg = 1-t
            neg_sign = torch.rand(batch_size, device=device, dtype=torch.float32) < 0.5
            t = torch.where(neg_sign, t_neg, t)
            dm = torch.where(neg_sign, -dm, dm)
            dn = torch.where(neg_sign, -dn, dn)
        return t, dm,dn


def sample_imm_trs(batch_size: int,  device: torch.device,  imm_k=None, verbose: bool = False):
            eps = 1e-5
            _t = torch.rand(batch_size, device=device, dtype=torch.float32)*(1-eps)+eps 
            _s = torch.rand(batch_size, device=device, dtype=torch.float32)*_t*(1-eps)+eps
            def compute_r(_t, _s, eta_max=160, eta_min=0, k=imm_k):#compute under the condition of t=0 is data, t=1 is noise
                alpha_t, sigma_t = 1-_t, _t 

                eta_t = sigma_t / alpha_t
                eps = (eta_max - eta_min) / 2**k
                eta = eta_t - eps
                _r = eta / (1.0 + eta)
                return torch.maximum(_s, _r)
            _r = compute_r(_t, _s)
            t = 1 - _t
            r =1 - _r
            s = 1 - _s
            if verbose:
                for i in range(batch_size):
                    print(f"t[i]: {t[i]:.3f}, r[i]: {r[i]:.3f}, s[i]: {s[i]:.3f}")
                    assert t[i] is not None and r[i] is not None and s[i] is not None, f"t[i]: {t[i]}, r[i]: {r[i]}, s[i]: {s[i]}"
            return t, r, s
            

@torch.no_grad()
def weight_bootstrap_losses(dt, progress, num_steps,weighting_mode="default", verbose=False):
    log2_num_step = np.log2(num_steps)
    assert dt.min() >=-1 and dt.max() <= 1
    if weighting_mode in ["default", "none"]:
        loss_weight = torch.ones((len(dt),)).to(dt.device)
    elif weighting_mode == "reverse":

       
        dt_abs = dt.abs()
        # Convert dt to log2 scale for easier manipulation
        # dt values: 1/128, 1/64, 1/32, ..., 1/2, 1
        # log2(1/dt): 7, 6, 5, ..., 1, 0
        log2_dt = -torch.log2(dt_abs)  # Convert to log scale
        
        # Create weights that favor small dt early and large dt late
        # When progress = 0: exp(-0 * 0 + 0 * 7) = 1 for small dt
        # When progress = 1: exp(-1 * 7 + 1 * 0) = small for small dt
        loss_weight = torch.exp((1-progress) * log2_dt + (progress) * (log2_num_step - log2_dt))
        
        # Normalize weights
        loss_weight = loss_weight / loss_weight.max()
        if verbose:
            for i in range(len(loss_weight)):
                print(f"progress={progress:.2f}, dt={dt[i]:.2f}, loss_weight={loss_weight[i]:.2f}")
    elif weighting_mode == "reverse_late_uniform":
        dt_abs = dt.abs()
        log2_dt = -torch.log2(dt_abs)  # Higher for small dt
        progress = torch.tensor(progress)
        if progress <= 0.7:
            # Use sigmoid for smooth transition centered at 0.7
            s0, k = 0.7, 0.07
            alpha = torch.sigmoid((progress - s0) / k)
            loss_weight = (1 - alpha) * log2_dt + alpha * torch.ones_like(log2_dt)
        else:
            loss_weight = torch.ones_like(log2_dt)


        # Normalize weights
        loss_weight = loss_weight / loss_weight.max()
        if verbose:
            for i in range(len(loss_weight)):
                print(f"progress={progress:.2f}, dt={dt[i]:.2f}, loss_weight={loss_weight[i]:.2f}")
    elif weighting_mode == "uniform":
        # At progress=0: small dt dominates (use log2_dt as weights)
        # At progress=0.5: uniform weights
        # Interpolate between the two

        dt_abs = dt.abs()
        log2_dt = -torch.log2(dt_abs)  # Higher for small dt

        # At progress=0: use log2_dt (small dt dominates)
        # At progress=0.5: use uniform (all ones)
        # Interpolate: (1 - 2*|progress-0.5|) goes from 0 at 0/1 to 1 at 0.5
        # But you want: progress=0 -> log2_dt, progress=0.5 -> uniform

        # Linear interpolation between log2_dt and uniform
        # progress=0: all log2_dt, progress=0.5: all ones
        # progress in [0, 0.5]: w = (1 - 2*progress) * log2_dt + (2*progress) * 1
        # progress in [0.5, 1]: just keep uniform (or you can extend logic)

        if progress <= 0.5:
            alpha = 2 * progress  # 0 at start, 1 at 0.5
            loss_weight = (1 - alpha) * log2_dt + alpha * torch.ones_like(log2_dt)
        else:
            # After 0.5, keep uniform
            loss_weight = torch.ones_like(log2_dt)

        # Normalize weights
        loss_weight = loss_weight / loss_weight.max()
        if verbose:
            for i in range(len(loss_weight)):
                print(f"progress={progress:.2f}, dt={dt[i]:.2f}, loss_weight={loss_weight[i]:.2f}")
    else:
        raise NotImplementedError(f"Weighting mode {weighting_mode} not implemented.")

    assert len(loss_weight) == len(dt)
    return loss_weight






# create batch, consisting of different timesteps and different dts(depending on total step sizes)
def create_targets_official(x1, x0, y, model, device, sc_kwargs, FORCE_T = -1, FORCE_DT = -1,CLASS_DROPOUT_PROB=1.0,NUM_CLASSES=-1, verbose=False, eps=1e-5):
    raise NotImplementedError("create_targets_official not implemented")
    assert len(x1) == len(x0) and len(x1) == len(y)
    bootstrap_every = sc_kwargs["bootstrap_every"]
    num_steps = sc_kwargs["num_steps"]
    flowloss_pyramid = sc_kwargs["flowloss_pyramid"]
    label_y = torch.zeros(x1.shape[0], device=device, dtype=torch.float32)
    current_bs = len(x1)
    model.eval()

    # 1. create step sizes dt
    bootstrap_bs = current_bs // bootstrap_every #=8
    log2_sections = int(math.log2(num_steps))
    # print(f"log2_sections: {log2_sections}")
    # print(f"bootstrap_batch_size: {bootstrap_batch_size}")

    def _sample_t_dt(_bs):
        dt_base = torch.repeat_interleave(log2_sections - 1 - torch.arange(log2_sections), _bs // log2_sections)
        # print(f"dt_base: {dt_base}")

        dt_base = torch.cat([dt_base, torch.zeros(_bs-dt_base.shape[0],)])
        # print(f"dt_base: {dt_base}")

        force_dt_vec = torch.ones(_bs) * FORCE_DT
        dt_base = torch.where(force_dt_vec != -1, force_dt_vec, dt_base).to(device)
        dt = 1 / (2 ** (dt_base)) # [1, 1/2, 1/8, 1/16, 1/32]
        # print(f"dt: {dt}")

        dt_base_bst = dt_base + 1
        dt_bst = dt / 2 # [0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 0.5]
        # print(f"dt_bootstrap: {dt_bootstrap}")

        # 2. sample timesteps t
        dt_sections = 2**dt_base

        # print(f"dt_sections: {dt_sections}")

        t = torch.cat([
            torch.randint(low=0, high=int(val.item()), size=(1,)).float()
            for val in dt_sections
            ]).to(device)
        
        # print(f"t[randint]: {t}")
        t = t / dt_sections
        # print(f"t[normalized]: {t}")
        
        force_t_vec = torch.ones(_bs, dtype=torch.float32).to(device) * FORCE_T
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(device)
        return t, dt_bst, dt_base_bst, dt, dt_base
    

    t, dt_bst, dt_base_bst, dt, dt_base = _sample_t_dt(bootstrap_bs)
    _onedt_bst = dt_bst
    t_full = t[:, None, None, None]
    
    if verbose:
        for i in range(bootstrap_bs):
            print(f"alg_type_bootstrap: t[i]: {t[i]}, dt_bootstrap[i]: {dt_bst[i]}, dt_base_bootstrap[i]: {dt_base_bst[i]}, dt[i]: {dt[i]}, dt_base[i]: {dt_base[i]}")

    # 3. generate bootstrap targets:
    x_1 = x1[:bootstrap_bs]
    _y_bst = y[:bootstrap_bs]
    x_0 = x0[:bootstrap_bs]

    # get dx at timestep t
    x_t = (1 - (1-eps) * t_full)*x_0 + t_full*x_1

    


    #with torch.no_grad():
    v_b1 = model(x_t, t, dt_bst, y=_y_bst)

    t2 = t + dt_bst
    x_t2 = x_t + dt_bst[:, None, None, None] * v_b1
    x_t2 = torch.clip(x_t2, -4, 4)
    
    #with torch.no_grad():
    v_b2 = model(x_t2, t2, dt_bst, y=_y_bst)

    v_target = (v_b1 + v_b2) / 2

    v_target = torch.clip(v_target, -4, 4)
    
    bst_v = v_target
    bst_dt = dt_base
    bst_t = t
    bst_xt = x_t
    bst_l = bst_labels

    # 4. generate flow-matching targets

    labels_dropout = torch.bernoulli(torch.full(label_y.shape, CLASS_DROPOUT_PROB)).to(device)
    labels_dropped = torch.where(labels_dropout.bool(), NUM_CLASSES, label_y)


    if flowloss_pyramid:
        t, onedt, onedt_base, twodt, twodt_base = _sample_t_dt(x1.shape[0])
        if True:#a small trick to make the network can directly set dt=1.0
            mask = torch.randint(0, 2, (x1.shape[0],), device=onedt.device, dtype=torch.bool)
            onedt = torch.where(mask, onedt, twodt)
        
    else: 
        # sample t(normalized)
        t = torch.randint(low=0, high=num_steps, size=(x1.shape[0],), dtype=torch.float32)
        # print(f"t: {t}")
        t /= num_steps
        # print(f"t: {t}")
        force_t_vec = torch.ones(x1.shape[0]) * FORCE_T
        # force_t_vec = torch.full((images.shape[0],), FORCE_T, dtype=torch.float32)
        t = torch.where(force_t_vec != -1, force_t_vec, t).to(device)
        # t_full = t.view(-1, 1, 1, 1)
        
        dt_flow = int(math.log2(num_steps))
        dt_base = (torch.ones(x1.shape[0], dtype=torch.int32) * dt_flow).to(device)
        onedt = 1 / (2 ** (dt_base))

    if verbose:
        for i in range(len(t)):
            print(f"alg_type_flow: t[i]: {t[i]}, onedt[i]: {onedt[i]}")
    t_full = t[:, None, None, None]    

    # sample flow pairs x_t, v_t
    x_0 = x0
    x_1 = x1
    x_t = (1 - (1 - eps) * t_full) * x_0 + t_full * x_1
    v_t = x_1 - (1 - eps) * x_0


    # 5. merge flow and bootstrap
    bs_shortcut = current_bs // bootstrap_every
    bs_flow = current_bs - bs_shortcut


    x_t = torch.cat([bst_xt, x_t[:bs_flow]], dim=0)
    t = torch.cat([bst_t, t[:bs_flow]], dim=0)
    v_t = torch.cat([bst_v, v_t[:bs_flow]], dim=0)
    onedt = torch.cat([_onedt_bst, onedt[:bs_flow]], dim=0)
    y = torch.cat([_y_bst, y[:bs_flow]], dim=0)
    labels_dropped = torch.cat([bst_l, labels_dropped[:bs_flow]], dim=0)
    model.train()
    return x_t, v_t, t, onedt, y, bootstrap_bs
    





def draw_schedule(weight_trend, dt_range, progress_list, subtitle_name=None,fig_file_name=None):
    """
    Draw a visualization of how loss weights change over training steps.
    
    Args:
        weight_trend: List of tensors containing loss weights for each training step
        dt_range: Tensor of dt values (timestep sizes)
        progress_list: List of training progress to visualize
        weighting_mode: String indicating the weighting mode used
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set up the plot
    plt.figure(figsize=(17, 8))
    
    # Number of dt values
    n_dt = len(dt_range)
    
    # Create subplots for selected training steps
    n_plots = len(progress_list)
    n_cols = 5
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Convert dt_range to actual timestep sizes for x-axis labels
    dt_labels = [f'1/{int(1/dt.item())}' for dt in dt_range]
    
    for idx, progress in enumerate(progress_list):
        plt.subplot(n_rows, n_cols, idx + 1)
        
        # Get weights for this step
        weights = weight_trend[idx].detach().cpu().numpy()
        
        # Create bar plot
        bars = plt.bar(range(n_dt), weights, alpha=0.7)
        
        # Customize plot
        plt.title(f'Training Progress {progress:.1%}')
        plt.xticks(range(n_dt), dt_labels, rotation=45)
        plt.ylabel('Probability/Weight')
        plt.xlabel('dt')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            x_pos = bar.get_x() + bar.get_width()/2.
            # Ensure positions are finite
            if np.isfinite(height) and np.isfinite(x_pos):
                plt.text(x_pos, height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
    
    # Adjust layout and add title
    plt.tight_layout()
    plt.suptitle(subtitle_name, y=1.02, fontsize=14)
    
    # Save the figure
    try:
        # Create directory if it doesn't exist
        #os.makedirs(os.path.dirname(fig_file_name), exist_ok=True)
        
        # Save with error handling
        plt.savefig(f"{fig_file_name}.png", bbox_inches='tight', dpi=300)
        print(f"Figure saved successfully to {fig_file_name}.png")
    except Exception as e:
        print(f"Error saving figure: {e}")
        # Try saving to current directory as fallback
        try:
            base_name = os.path.basename(fig_file_name)
            plt.savefig(f"{base_name}.png", bbox_inches='tight', dpi=300)
            print(f"Figure saved to current directory as {base_name}.png")
        except Exception as e2:
            print(f"Failed to save figure: {e2}")
    
    plt.close()
    


def draw_loss_weight(weighting_mode="uniform",  num_steps=128):
    
    dt_range = torch.tensor([1/(2**i) for i in range(7, -1, -1)])
    weight_trend = []
    progress_list=[i for i in np.linspace(0, 0.9, 10)]
    for _current_progress in progress_list:
        loss_weight = weight_bootstrap_losses(dt=dt_range, weighting_mode=weighting_mode, progress=_current_progress, num_steps=num_steps,verbose=True)
        weight_trend.append(loss_weight)

    draw_schedule(weight_trend, dt_range, progress_list=progress_list,subtitle_name=f'Loss Weights Evolution ({weighting_mode} mode)', fig_file_name=f'loss_weight_{weighting_mode}')

def draw_dt_annealing_schedule(annealing_mode="reverse",  time_steps=128, bs=2048, dt_negative=False,verbose=False,etazero=False):
    dt_range = torch.tensor([1/(2**i) for i in range(7, -1, -1)])
    dt_annealing_trend = []
    progress_list=[i for i in np.linspace(0, 0.99, 20)]
    for _current_progress in progress_list:
        t,  twodt, twodt_base, dt, dt_base  = sample_delta_eta(batch_size=bs, device=torch.device("cpu"), etazero=etazero,annealing_mode=annealing_mode,  train_progress=_current_progress, dt_negative=dt_negative,verbose=verbose)
        # Calculate histogram of dt values
        
        assert t.min() >= 0 and t.max() <= 1, f"t.min(): {t.min()}, t.max(): {t.max()}"
        dt_counts = torch.zeros(len(dt_range))
        for i, target_dt in enumerate(dt_range):
            dt_counts[i] = (dt == target_dt).sum() 
        dt_counts = dt_counts / dt_counts.sum()
        dt_annealing_trend.append(dt_counts)

    draw_schedule(dt_annealing_trend, dt_range, progress_list, subtitle_name=f'DT Annealing Schedule ({annealing_mode} mode)', fig_file_name=f'v2_dt_annealing_schedule_{annealing_mode}')


def draw_imm_dmdn_schedule(annealing_mode="continuous",  time_steps=128, bs=2048, dt_negative=False,verbose=False):
    dt_range = torch.tensor([1/(2**i) for i in range(7, -1, -1)])
    dt_annealing_trend = []
    progress_list=[i for i in np.linspace(0, 0.99, 20)]
    for _current_progress in progress_list:
        t, dm, dn  = sample_immfm_dmdn(batch_size=bs, device=torch.device("cpu"), annealing_mode=annealing_mode,  train_progress=_current_progress, dt_negative=dt_negative,num_steps=time_steps, verbose=verbose)
        # Calculate histogram of dt values
    


def draw_dt_eta():
    import matplotlib.pyplot as plt
    t, dt2, _, dt1, _ =sample_delta_eta(name='fm',batch_size=1024, device=torch.device("cpu"), train_progress=0.99, verbose=True)
    plt.figure(figsize=(17, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(t, dt1, label='dt1')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(t, dt2, label='dt2')
    plt.legend()
    plt.savefig('dt_eta.png', dpi=300, bbox_inches='tight')
    plt.close()


def draw_dt_eta_mapping():
    import matplotlib.pyplot as plt
    def eta_t(t):
        return t/(1-t)
    
    t = torch.linspace(0, 1, 1000)
    eta = eta_t(t)
    plt.figure(figsize=(17, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(t, eta, label='eta')
    plt.legend()

    eta_trimmed = eta[eta > 160]
    t_trimmed = t[eta > 160]
    eta_trimmed = eta_trimmed/2**9
    plt.subplot(1, 2, 2)
    plt.scatter(eta_trimmed, t_trimmed, label='t')
    plt.legend()
    plt.savefig('dt_eta_mapping.png', dpi=300, bbox_inches='tight')
    plt.close()




def edm_rho_in_fm(N=10):
    def get_t_schedule(N, t_min=0.0, t_max=1.0, rho=7):
        t_rho = lambda t: t**(1/rho)
        inv_t_rho = lambda x: x**rho
        A = t_rho(t_max)
        B = t_rho(t_min)
        ts = [inv_t_rho(B + (i / (N - 1)) * (A-B)) for i in range(N)]
        return ts
    t = torch.linspace(0, 1, N)
    ts = get_t_schedule(N, t_min=0.0, t_max=1.0, rho=7)
    plt.figure(figsize=(17, 8))
    plt.scatter(t, ts, label='t')
    plt.legend()
    plt.savefig('edm_rho_in_fm.png', dpi=300, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":


    if False:
        import sys
        sys.path.append("..")
        from models.dit_shortcut import DiT_S_4
        model = DiT_S_4(use_dt_adaptor=True,use_shortcut=True)
        bs = 64
        x = torch.randn(bs, 4, 32, 32)
        y = torch.randn(bs, 4, 32, 32)
        x0 = torch.randn(bs, 4, 32, 32)
        annealing_mode = "none"
        weighting_mode = "reverse"
        dt_negative = False
        train_progress=0
        num_steps = 16
        flowloss_pyramid = False
        verbose = True
        alg_type = "immfm"
        use_ema_for_bootstrap = True
        
    elif False:
        num_steps = 128
        draw_loss_weight(num_steps=num_steps,weighting_mode="reverse_late_uniform")
    elif True:
        draw_dt_annealing_schedule(annealing_mode="beta_0.7", bs=16, dt_negative=False,verbose=True,etazero=True)
    elif False:
        draw_imm_dmdn_schedule(annealing_mode="imm", time_steps=128, bs=16, dt_negative=False,verbose=True)
    elif False:
        sample_imm_trs(batch_size=256, device=torch.device("cpu"), verbose=True)
    elif False:
        draw_dt_eta()
    elif False:
        draw_dt_eta_mapping()
    elif False:
        edm_rho_in_fm()


        