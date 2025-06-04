import torch
import torch.nn as nn
import torch.nn.functional as NF

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

import math


def pad_warp(x):
    H,W,C = x.shape
    ret = torch.zeros(H+2,W+2,C,device=x.device)
    ret[1:-1,1:-1] = x
    ret[0,1:-1] = x[-1]
    ret[-1,1:-1] = x[0]
    ret[1:-1,0] = x[:,-1]
    ret[1:-1,-1] = x[:,0]
    
    ret[0,0] = x[-1,-1]
    ret[0,-1] = x[-1,0]
    ret[-1,0] = x[0,-1]
    ret[-1,-1] = x[0,0]
    return ret


def Nx(normal):
    H,W,C = normal.shape
    normal_pad = pad_warp(normal)
    
    kernel = torch.tensor([[0,0,0],
                           [-0.5,0.0,0.5],
                          [0,0,0]],device=normal.device)
    
    ret = NF.conv2d(normal_pad.permute(2,0,1).unsqueeze(1),
                   kernel.reshape(1,1,3,3))
    ret = ret.reshape(C,H,W).permute(1,2,0)
    return ret

def Ny(normal):
    H,W,C = normal.shape
    normal_pad = pad_warp(normal)
    
    kernel = torch.tensor([[0,0,0],
                           [-0.5,0.0,0.5],
                          [0,0,0]],device=normal.device).T
    
    ret = NF.conv2d(normal_pad.permute(2,0,1).unsqueeze(1),
                   kernel.reshape(1,1,3,3))
    ret = ret.reshape(C,H,W).permute(1,2,0)
    return ret

def Nxy(normal):
    H,W,C = normal.shape
    normal_pad = pad_warp(normal)
    
    kernel = torch.tensor([
        [0.5,-0.5,0],
        [-0.5,1.0,-0.5],
        [0,-0.5,0.5]
    ])
    
    ret = NF.conv2d(normal_pad.permute(2,0,1).unsqueeze(1),
                   kernel.reshape(1,1,3,3))
    ret = ret.reshape(C,H,W).permute(1,2,0)
    return ret

def compute_coeff(x):
    """x:HxWxCxK"""
    A_inv = torch.tensor(
        [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
         [-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0],
         [9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1],
         [-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1],
         [2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0],
         [-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1],
         [4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1]],
        device=x.device).float()
    return torch.einsum('ij,hwcj->hwci',A_inv,x)

def bicubic_coeff(normal):
    nx = pad_warp(Nx(normal))
    ny = pad_warp(Ny(normal))
    nxy = pad_warp(Nxy(normal))
    n = pad_warp(normal)
    
    H,W,C = normal.shape
    x = torch.stack([
        n[1:-1,1:-1],n[1:-1,2:],n[2:,1:-1],n[2:,2:],
        nx[1:-1,1:-1],nx[1:-1,2:],nx[2:,1:-1],nx[2:,2:],
        ny[1:-1,1:-1],ny[1:-1,2:],ny[2:,1:-1],ny[2:,2:],
        nxy[1:-1,1:-1],nxy[1:-1,2:],nxy[2:,1:-1],nxy[2:,2:]
    ],-1)
    x = compute_coeff(x).reshape(H,W,C,4,4)#.permute(0,1,2,4,3)
    return x
    

def getN(coeff,uv):
    """uv: Bx2"""
    H,W,C,_,_ = coeff.shape
    v,u = uv[...,0]*H,uv[...,1]*W
    u0,v0 = u.floor().long(),v.floor().long()
    u,v = u-u0,v-v0
    u0,v0 = u0%H,v0%W
    
    coeff = coeff.reshape(H*W,C,4,4)[u0*W+v0]
    i,j = torch.meshgrid(torch.arange(4,device=coeff.device),
                         torch.arange(4,device=coeff.device))
    
    weight = u[:,None,None].pow(i[None])*v[:,None,None].pow(j[None])
    return torch.einsum('bcij,bij->bc',coeff,weight)
    
def getAABB(normal,sigma_r):
    nmin = (normal-3*sigma_r).min()
    nmax = (normal+3*sigma_r).max()
    return nmin,nmax

def sampleGauss2d(u):
    r1,r2 = u[...,0],u[...,1]
    tmp = (-2*r1.clamp_min(1e-8).log()).sqrt()
    x = tmp*torch.cos(2*math.pi*r2)
    y = tmp*torch.sin(2*math.pi*r2)
    return torch.stack([x,y],-1)


class PositionNormal(nn.Module):
    def __init__(self,normal,sigma_r_min=0.003):
        super(PositionNormal,self).__init__()
        if type(normal) is str:
            normal = cv2.imread(normal,-1)[...,0:1]
            normal = torch.from_numpy(normal).float()
            
        if normal.shape[-1] == 1:
            # height field
            nx = Nx(normal)
            ny = Ny(normal)
            nz = (1-nx*nx-ny*ny).relu().sqrt()
            normal = torch.cat([nx,ny,nz],-1)
            normal = NF.normalize(normal,dim=-1)
        normal = normal[...,:2]
        nmin,nmax = getAABB(normal,sigma_r_min)
        self.sigma_r_min = sigma_r_min
        #nmin,nmax = -1,1
        
        self.nmin = nmin
        self.nmax = nmax
        self.register_buffer('normal_coeff',bicubic_coeff(normal)[:,:,:2])
    
    def forward(self,u,s,sigma_r=None):
        '''s,n: Bx3 -1,1'''
        n = getN(self.normal_coeff,u*0.5+0.5)
        sigma_r = sigma_r if sigma_r is not None else self.sigma_r_min
        ndf = 1.0/(2*math.pi*sigma_r)*torch.exp(-0.5*((n-s)/sigma_r).pow(2).sum(-1))
        return ndf
    
    def sample(self,u,sample2,sigma_r=None):
        """ 
        u: Bx2 position from [-1,1]
        sample2: BxSx2 unform samples from [0,1]
        """
        # stratified + uniform sampling
        sigma_r = sigma_r if sigma_r is not None else self.sigma_r_min
        n = getN(self.normal_coeff,u*0.5+0.5)
        s_g = sampleGauss2d(sample2)*sigma_r + n[:,None]
        s_u = sample2*(self.nmax-self.nmin)+self.nmin
        
        s = torch.cat([s_g,s_u],1)
        ndf = 1.0/(2*math.pi*sigma_r)\
            * torch.exp(-0.5*((n[:,None]-s)/sigma_r).pow(2).sum(-1))
        return s,ndf
    
    def binning(self,center,
                sigma_p=36//6,sigma_r=None,
                SPP=10000000,res=256,
                batch_size=4*10240):
        """center: -1,1 """
        H = self.normal_coeff.shape[0]
        center = (center*0.5+0.5)*H
        N = int(math.sqrt(SPP))
        device = center.device
        
        sigma_r = sigma_r if sigma_r is not None else self.sigma_r_min
        inds = torch.zeros(N*N,dtype=torch.long,device=device)
        i,j = torch.meshgrid(torch.arange(N,device=device),
                             torch.arange(N,device=device))
        ij = torch.stack([j,i],-1).reshape(-1,2).float()
        
        nmin,nmax = self.nmin,self.nmax
        
        for b in range(math.ceil(N*N*1.0/batch_size)):
            b0 = b*batch_size
            b1 = min(b0+batch_size,N*N)

            u0 = (ij[b0:b1]+torch.rand(b1-b0,2,device=device))/N
            g = sampleGauss2d(u0)
            u = (g*sigma_p+center)/H
            n = getN(self.normal_coeff,u)
            n += sigma_r*torch.randn(b1-b0,2,device=device)
            ni = (n-nmin)/(nmax-nmin)
            ni = (ni*res).long().clamp(0,res-1)
            inds[b0:b1] = ni[...,0] + ni[...,1]*res
        
        GNDF = torch.zeros(res*res,device=device)
        GNDF.scatter_add_(0,inds,torch.ones_like(inds).float())
        GNDF /= GNDF.sum()
        GNDF = GNDF.reshape(res,res)
        return GNDF
    
    def query(self,u,s,sigma_p,sigma_r=None,SPP=10000):
        """u,s:Bx2 in [-1,1]"""
        sigma_r = sigma_r if sigma_r is not None else self.sigma_r_min
        device = u.device
        B = u.shape[0]
        H = self.normal_coeff.shape[0]
        N = int(math.sqrt(SPP))
        sample2 = torch.meshgrid(torch.arange(N,device=device),torch.arange(N,device=device))
        sample2 = torch.stack([sample2[1],sample2[0]],-1).float().reshape(1,-1,2)
        sample2 = sample2+torch.rand(B,N*N,2,device=device)
        sample2 /= N
        
        
        u = (u*0.5+0.5)
        du = sampleGauss2d(sample2)*sigma_p/H
        
        n = getN(self.normal_coeff,(du+u[:,None]).reshape(-1,2)).reshape(B,-1,2)
        ndf = 1/(2*math.pi*sigma_r)*torch.exp(-0.5*((n-s[:,None])/sigma_r).pow(2).sum(-1))
        
        return ndf.mean(1)