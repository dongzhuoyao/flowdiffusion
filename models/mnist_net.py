# Borrowing this implentation from https://github.com/bot66/MNISTDiffusion
# 
# Added the following modifications:
# - Using Positional Embeddings instead of learned embeddings (Which only work in discrete space)
# - Using AdaGN with PixelNorm like sCM recommends to inject time embeddings
# - Added logvar output
# - Using GroupNorm instead of BatchNorm: I found BatchNorm does not play nicely with JVP
#
# MIT License
# Copyright (c) 2022 Guocheng Tan
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
# associated documentation files (the "Software"), to deal in the Software without restriction, including 
# without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        half_dim = num_channels // 2
        emb = math.log(10) / (half_dim - 1)
        self.register_buffer('freqs', torch.exp(torch.arange(half_dim) * -emb))
        
    def forward(self, x):
        y = x.to(torch.float32)
        y = y.outer(self.freqs.to(torch.float32))
        y = torch.cat([torch.sin(y), torch.cos(y)], dim=1)
        return y.to(x.dtype)

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, act_fn=nn.SiLU()):
        super().__init__()
        if hidden_features is None:
            hidden_features = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act_fn = act_fn
        self.fc2 = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


class ChannelShuffle(nn.Module):
    def __init__(self,groups):
        super().__init__()
        self.groups=groups
    def forward(self,x):
        n,c,h,w=x.shape
        x=x.view(n,self.groups,c//self.groups,h,w) # group
        x=x.transpose(1,2).contiguous().view(n,-1,h,w) #shuffle
        
        return x

class ConvBnSiLu(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.gn = nn.GroupNorm(num_groups=max(1, out_channels//8), num_channels=out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.silu(x)
        return x

class ResidualBottleneck(nn.Module):
    '''
    shufflenet_v2 basic unit(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.branch1=nn.Sequential(nn.Conv2d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.GroupNorm(num_groups=max(1, (in_channels//2)//8), num_channels=in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels//2,in_channels//2,1,1,0),
                                    nn.Conv2d(in_channels//2,in_channels//2,3,1,1,groups=in_channels//2),
                                    nn.GroupNorm(num_groups=max(1, (in_channels//2)//8), num_channels=in_channels//2),
                                    ConvBnSiLu(in_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle=ChannelShuffle(2)

    def forward(self,x):
        x1,x2=x.chunk(2,dim=1)
        x=torch.cat([self.branch1(x1),self.branch2(x2)],dim=1)
        x=self.channel_shuffle(x) #shuffle two branches

        return x

class ResidualDownsample(nn.Module):
    '''
    shufflenet_v2 unit for spatial down sampling(https://arxiv.org/pdf/1807.11164.pdf)
    '''
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.branch1=nn.Sequential(nn.Conv2d(in_channels,in_channels,3,2,1,groups=in_channels),
                                    nn.GroupNorm(num_groups=max(1, in_channels//8), num_channels=in_channels),
                                    ConvBnSiLu(in_channels,out_channels//2,1,1,0))
        self.branch2=nn.Sequential(ConvBnSiLu(in_channels,out_channels//2,1,1,0),
                                    nn.Conv2d(out_channels//2,out_channels//2,3,2,1,groups=out_channels//2),
                                    nn.GroupNorm(num_groups=max(1, (out_channels//2)//8), num_channels=out_channels//2),
                                    ConvBnSiLu(out_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle=ChannelShuffle(2)

    def forward(self,x):
        x=torch.cat([self.branch1(x),self.branch2(x)],dim=1)
        x=self.channel_shuffle(x) #shuffle two branches

        return x

class TimeMLP(nn.Module):
    '''
    naive introduce timestep information to feature maps with mlp and add shortcut
    '''
    def __init__(self,embedding_dim,hidden_dim,out_dim):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(embedding_dim,hidden_dim),
                                nn.SiLU(),
                               nn.Linear(hidden_dim,out_dim))
        self.act=nn.SiLU()
    def forward(self,x,t):
        emb = t  # t is already embedded
        c = self.mlp(emb) + 1
        
        # PixelNorm as in sCM
        c = c / torch.sqrt(torch.mean(c ** 2, dim=1, keepdim=True) + 1e-8)
        
        # Inject time conditioning
        x = x * c.unsqueeze(2).unsqueeze(3).to(x.dtype)
        
        return self.act(x)
    
class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,out_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=out_channels,out_dim=out_channels//2)
        self.conv1=ResidualDownsample(out_channels//2,out_channels)
    
    def forward(self,x,t=None):
        x_shortcut=self.conv0(x)
        if t is not None:
            x=self.time_mlp(x_shortcut,t)
        x=self.conv1(x)

        return [x,x_shortcut]
        
class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,in_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=in_channels,out_dim=in_channels//2)
        self.conv1=ResidualBottleneck(in_channels//2,out_channels//2)

    def forward(self,x,x_shortcut,t=None):
        x = nn.functional.interpolate(x, size=x_shortcut.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x_shortcut], dim=1)
        x = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x, t)
        x = self.conv1(x)
        return x        

class Unet(nn.Module):
    '''
    simple unet design without attention
    '''
    def __init__(self,time_embedding_dim=256,in_channels=3,out_channels=2,base_dim=32,dim_mults=[2,2,4,4]):
        super().__init__()
        assert isinstance(dim_mults,(list,tuple))
        assert base_dim%2==0 

        channels=self._cal_channels(base_dim,dim_mults)

        self.init_conv=ConvBnSiLu(in_channels,base_dim,3,1,1)
        self.time_embedding=PositionalEmbedding(time_embedding_dim)
        self.time_mlp=MLP(time_embedding_dim,time_embedding_dim,time_embedding_dim)
        self.dt_embedding=PositionalEmbedding(time_embedding_dim)
        self.dt_mlp=MLP(time_embedding_dim,time_embedding_dim,time_embedding_dim)
        self.encoder_blocks=nn.ModuleList([EncoderBlock(c[0],c[1],time_embedding_dim) for c in channels])
        self.decoder_blocks=nn.ModuleList([DecoderBlock(c[1],c[0],time_embedding_dim) for c in channels[::-1]])
    
        self.mid_block=nn.Sequential(*[ResidualBottleneck(channels[-1][1],channels[-1][1]) for i in range(2)],
                                        ResidualBottleneck(channels[-1][1],channels[-1][1]//2))
        
        self.logvar_linear = nn.Linear(time_embedding_dim, 1)

        self.final_conv=nn.Conv2d(in_channels=channels[0][0]//2,out_channels=out_channels,kernel_size=1)

    def forward_without_cfg(self,*args,**kwargs):
        return self.forward(*args,**kwargs)
        

    def forward(self,x,t,dt, y, return_logvar=False, **kwargs):
        if len(x.shape) == len(t.shape):
            t = t.squeeze(-1).squeeze(-1).squeeze(-1)
        if len(x.shape) == len(dt.shape):
            dt = dt.squeeze(-1).squeeze(-1).squeeze(-1)

        x=self.init_conv(x)
        
        t=self.time_mlp(self.time_embedding(t))
        dt = self.dt_mlp(self.dt_embedding(dt))
        t = t + dt
        encoder_shortcuts=[]
        for encoder_block in self.encoder_blocks:
            x,x_shortcut=encoder_block(x,t)
            encoder_shortcuts.append(x_shortcut)
        x=self.mid_block(x)
        encoder_shortcuts.reverse()
        for decoder_block,shortcut in zip(self.decoder_blocks,encoder_shortcuts):
            x=decoder_block(x,shortcut,t)
        x=self.final_conv(x)
        
        if return_logvar:
            logvar = self.logvar_linear(t)
            return x, logvar
        else:
            return x


    def _cal_channels(self,base_dim,dim_mults):
        dims=[base_dim*x for x in dim_mults]
        dims.insert(0,base_dim)
        channels=[]
        for i in range(len(dims)-1):
            channels.append((dims[i],dims[i+1])) # in_channel, out_channel

        return channels

if __name__=="__main__":
    bs =10
    x=torch.randn(bs,4,28,28)
    t=torch.randn(bs,1,1,1)
    dt=torch.randn(bs,1,1,1)
    model=Unet(256, in_channels=4, out_channels=4, base_dim=64, dim_mults=[1, 2, 4])
    y=model(x,t,dt,y=None)
    print(y.shape)