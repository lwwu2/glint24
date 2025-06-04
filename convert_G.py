import cv2
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np
import torch
import torch.nn.functional as NF
import math

from argparse import ArgumentParser
from tqdm import tqdm

from utils.normal_map import PositionNormal,Nx,Ny
from utils.cuda import project_triangle


""" 
generate GGX projected area approximation
currently need to run twice:
1. generate alpha and tangent map using this script.
2. run mitsuba to generate corresponding .mip file
3. overwrite .mip file by run this script again.
"""

def triangle_area(p1,p2,p3):
    """ triangle area inside vertex p1,p2,p3"""
    p13 = p1-p3
    p23 = p2-p3
    return p13[...,0]*p23[...,1]-p23[...,0]*p13[...,1]



if __name__ == '__main__':
    
    parser = ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--mip',dest="mip",action='store_true')
    parser.add_argument('--normal',dest="normal",action='store_true')
    parser.set_defaults(normal=False)
    parser.set_defaults(mip=False)
    args = parser.parse_args()
    
    INPUT = args.input # path to normal map
    filename = INPUT.split('/')[-1].split('.')[0]
    OUTPUT = args.output # path to output folder
    mip = args.mip # whether overwrite mip file
    read_normal = args.normal # is heigh field or normal map
    
    
    device = torch.device(0) # use GPU device 0 for acceleration
    
    # read input normal map
    if not read_normal:
        height = cv2.imread(INPUT,-1)
        height = torch.from_numpy(height[...,0:1]).float()
        nx = Nx(height)
        ny = Ny(height)
    else:
        nxy = cv2.imread(INPUT,-1)[...,[2,1,0]]
        nxy = torch.from_numpy(nxy).float()
        nx,ny = nxy[...,0:1],nxy[...,1:2]

    nz = (1-nx*nx-ny*ny).relu().sqrt()
    normal = torch.cat([nx,ny,nz],-1)

    
    if normal.shape[0] == 1024:
        normal_patch = torch.cat([normal,normal[:,0:1]],1)
        normal_patch = torch.cat([normal_patch,normal_patch[0:1]],0)[...,:2]
    else:
        normal_patch = normal[...,:2]

    verts = torch.stack([
        normal_patch[:-1,:-1],normal_patch[:-1,1:],normal_patch[1:,:-1],
        normal_patch[1:,:-1],normal_patch[:-1,1:],normal_patch[1:,1:]],
        -2).reshape(-1,3,2)
    
    
    # build mip-map of tangent frame and alpha
    rots = []
    alphas = []
    kk = 1
    
    # sample points in grazing angle
    phi = torch.linspace(0,2*math.pi,360,device=device)
    R = torch.stack([torch.cos(phi),-torch.sin(phi),
                     torch.sin(phi),torch.cos(phi)],-1).reshape(-1,2,2)
    x,y = torch.cos(phi),torch.sin(phi)
    verts= verts.to(device)

    offset = [0]
    H = normal.shape[0]
    print("bake GGX parameters")
    while H!=2:
        offset.append(offset[-1]+H*H)
        verts_mu = verts.reshape(-1,3,2)
        area = triangle_area(verts_mu[:,0],verts_mu[:,1],verts_mu[:,2]).abs()
        verts3 = verts_mu[:,0]

        G_est = torch.zeros(360,H*kk,H*kk,device=device)
        for i in tqdm(range(len(x))):
            p012 = verts_mu@R[i].reshape(1,2,2) # convert to canonical setting
            proj = project_triangle(p012[:,0],p012[:,1],p012[:,2],0)
            proj = torch.where(area>1e-6,(proj/area),(verts3[...,0]*x[i]+verts3[...,1]*y[i]).relu()/2)
            G_est[i] = proj.reshape(1024,1024,2).sum(-1).cpu()
        G_est = NF.avg_pool2d(G_est[:,None],kk,kk)[:,0]

        A = torch.stack([x*x,y*y,x*y],-1)
        B = (2*G_est.reshape(360,-1)).pow(2)
        coeff = torch.linalg.lstsq(A.cpu(),B.cpu())[0].to(device)
        Q,S,V = torch.stack([coeff[0],coeff[2]/2,
                             coeff[2]/2,coeff[1]],-1).reshape(-1,2,2).svd()

        Q = Q.permute(0,2,1).reshape(H,H,2,2)[:,:,:,0]
        axy = S.sqrt().reshape(H,H,2)

        rots.append(torch.cat([Q,torch.zeros_like(Q[...,0:1])],-1).cpu())
        alphas.append(torch.cat([axy,torch.zeros_like(axy[...,0:1])],-1).cpu())

        kk = kk<<1
        H//=2

    
    # store file
    
    if not mip: # first run, only store alpha and tangent frame
        print("save parameters")
        os.makedirs(f'{OUTPUT}/{filename}/',exist_ok=True)
        cv2.imwrite(f'{OUTPUT}/{filename}/alpha.exr',alphas[0][...,[2,1,0]].numpy())
        cv2.imwrite(f'{OUTPUT}/{filename}/rot.exr',rots[0][...,[2,1,0]].numpy())
    else: # second run, overwrite mitsuba .mip file
        print("overwrite mip-map")
        pad=64 # .mip file padding
        blockSize=4 # mitsuba block arry convention
        
        if os.path.exists(f'{OUTPUT}/{filename}/alpha.mip'):    
            mip_file = np.fromfile(f'{OUTPUT}/{filename}/alpha.mip',dtype='float16')
            for i in range(len(alphas)):
                block = alphas[i]
                H = block.shape[0]
                block = block.reshape(H//blockSize,blockSize,H//blockSize,blockSize,3)
                block = block.permute(0,2,1,3,4).reshape(-1,3).numpy().astype(np.float32)
                mip_file[pad+offset[i]*3:pad+offset[i+1]*3] = block.reshape(-1)
            mip_file.tofile(f'{OUTPUT}/{filename}/alpha.mip')
        else:
            print("Need to run mitsuba first to create alpha.mip file.")

        if os.path.exists(f'{OUTPUT}/{filename}/rot.mip'):    
            mip_file = np.fromfile(f'{OUTPUT}/{filename}/rot.mip',dtype='float16')
            for i in range(len(rots)):
                block = rots[i]
                H = block.shape[0]
                block = block.reshape(H//blockSize,blockSize,H//blockSize,blockSize,3)
                block = block.permute(0,2,1,3,4).reshape(-1,3).numpy().astype(np.float32)
                mip_file[pad+offset[i]*3:pad+offset[i+1]*3] = block.reshape(-1)
            mip_file.tofile(f'{OUTPUT}/{filename}/rot.mip')
        else:
            print("Need to run mitsuba first to create rot.mip file.")