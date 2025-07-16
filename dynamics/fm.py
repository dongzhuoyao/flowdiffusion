from torchdiffeq import odeint
import torch
import torch.nn.functional as F
import torch.optim as optim

import math
import os
from typing import Any, Dict, List, Optional

from dynamics.shortcut_utils import shortcut_loss

def sample_t_fn(x1):
    t = torch.rand(len(x1), device=x1.device)
    t_vector = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return t,t_vector

class FlowMatching:
    def __init__(self, with_shortcut=False, eps=1e-4):
        self.with_shortcut = with_shortcut
        print(f"with_shortcut: {with_shortcut}")
        self.eps = eps

    def _fm_loss(self, model, x1, **model_kwargs):
        x0 = torch.randn_like(x1)
        t,t_vector = sample_t_fn(x1)
        x_t =   (1 - t_vector) * x0 + t_vector * x1
        v_t_pred = model(x_t, t, dt=None, **model_kwargs)
        u_t = x1 - x0

        loss = F.mse_loss(v_t_pred, u_t)
        return {"loss": loss}



    def training_losses(self, model, ema_model, x1, sc_kwargs,verbose=False, **model_kwargs):
        if self.with_shortcut:
            loss_dict = shortcut_loss(model=model, ema_model=ema_model, x1=x1, verbose=verbose, sc_kwargs=sc_kwargs, **model_kwargs)
            return loss_dict
        else:
            loss_dict =  self._fm_loss(model=model, x1=x1, **model_kwargs)
            return loss_dict



    @torch.no_grad()
    def sample_fn(self, z, model_fn, step_num, **model_kwargs):
        dt = 1.0/step_num
        dt_vec = torch.ones(len(z),device=z.device)*dt
        device = z.device
        verbose = model_kwargs.get("verbose",False)
        if verbose:
            print(f"dt: {dt}, z.shape: {z.shape}, z.min: {z.min()}, z.max: {z.max()}")

        def sample_fn_wrapper(model,*args,**kwargs):
             res = model(*args,**kwargs)
             if isinstance(res,tuple):#if is repa model, return only the first element
                return res[0]
             else:
                return res
        
        if self.with_shortcut:
                x_t = z 
                for i in range(step_num):
                    t = i*1.0/step_num
                    t_vec = torch.ones(len(x_t),device=device)*t

                    x_t = x_t + dt * sample_fn_wrapper(model_fn,x_t, t_vec, dt_vec, **model_kwargs)
                    if verbose:
                        print(f"i,x_t: {i},{x_t.shape},{x_t.min()},{x_t.max()}")
                return x_t
        else: # FM
                x_t = z 
                for i in range(step_num):
                    t = i*1.0/step_num
                    t_vec = torch.ones(len(x_t),device=device)*t
                    x_t = x_t + dt * sample_fn_wrapper(model_fn,x_t, t_vec, dt=None, **model_kwargs)
                return x_t
       
            


if __name__ == "__main__":
    pass 