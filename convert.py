import cv2
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import numpy as np
import torch
import torch.nn.functional as NF
import math

from argparse import ArgumentParser

from utils.normal_map import PositionNormal,Nx,Ny


""" convert normal map to the bounding box and cluster hierarchy"""

def triangle_area(p1,p2,p3):
    """ triangle area inside vertex p1,p2,p3"""
    p13 = p1-p3
    p23 = p2-p3
    return p13[...,0]*p23[...,1]-p23[...,0]*p13[...,1]



def triangle2Matrix(s1,s2,s3,A,B1,B2,LU0,LU1):
    """
    build least square matrix for current triangle
    Args:
        s1,s2,s3: Nx2
        A,B1,B2: N
        LU0: upper or lower normal map triangle
        LU1: upper or lower cluster triangle
    Return:
        M: Nx3x3
        b: Nx3x2
        c: Nx2
    """
    N = s1.shape[0]
    if LU0 == 'upper':
        if LU1 == 'upper':
            M  = torch.stack([
                (1/6)*(6*A**2 - 4*A*(3*B1 + 3*B2 + 2) + 6*B1**2 + 12*B1*B2 + 8*B1 + 6*B2**2 + 8*B2 + 3)/A**2, ((1/3)*A*(3*B1 + 1) - B1**2 - B1*B2 - B1 - 1/3*B2 - 1/4)/A**2, ((1/3)*A*(3*B2 + 1) - B1*B2 - 1/3*B1 - B2**2 - B2 - 1/4)/A**2, 
                ((1/3)*A*(3*B1 + 1) - B1**2 - B1*B2 - B1 - 1/3*B2 - 1/4)/A**2, (1/6)*(6*B1**2 + 4*B1 + 1)/A**2, (B1*B2 + (1/3)*B1 + (1/3)*B2 + 1/12)/A**2, 
                ((1/3)*A*(3*B2 + 1) - B1*B2 - 1/3*B1 - B2**2 - B2 - 1/4)/A**2, (B1*B2 + (1/3)*B1 + (1/3)*B2 + 1/12)/A**2, (1/6)*(6*B2**2 + 4*B2 + 1)/A**2
            ],-1).reshape(N,3,3)
            A,B1,B2 = A[...,None],B1[...,None],B2[...,None]
            b = torch.stack([
                (1/12)*(2*s1*(-2*A + 2*B1 + 2*B2 + 1) + s2*(-4*A + 4*B1 + 4*B2 + 3) + s3*(-4*A + 4*B1 + 4*B2 + 3))/A, 
                (1/12)*(-s1*(4*B1 + 1) - 2*s2*(2*B1 + 1) - s3*(4*B1 + 1))/A, 
                (1/12)*(-s1*(4*B2 + 1) - s2*(4*B2 + 1) - 2*s3*(2*B2 + 1))/A
            ],-2).reshape(N,3,2)
        elif LU1 == 'lower':
            M = torch.stack([
                (1/6)*(6*A**2 - 4*A*(3*B1 + 3*B2 + 4) + 6*B1**2 + 12*B1*B2 + 16*B1 + 6*B2**2 + 16*B2 + 11)/A**2, 
                ((1/3)*A*(3*B1 + 2) - B1**2 - B1*B2 - 2*B1 - 2/3*B2 - 11/12)/A**2, ((1/3)*A*(3*B2 + 2) - B1*B2 - 2/3*B1 - B2**2 - 2*B2 - 11/12)/A**2, 
                ((1/3)*A*(3*B1 + 2) - B1**2 - B1*B2 - 2*B1 - 2/3*B2 - 11/12)/A**2, (1/6)*(6*B1**2 + 8*B1 + 3)/A**2, (1/12)*(12*B1*B2 + 8*B1 + 8*B2 + 5)/A**2, 
                ((1/3)*A*(3*B2 + 2) - B1*B2 - 2/3*B1 - B2**2 - 2*B2 - 11/12)/A**2, (1/12)*(12*B1*B2 + 8*B1 + 8*B2 + 5)/A**2, (1/6)*(6*B2**2 + 8*B2 + 3)/A**2
            ],-1).reshape(N,3,3)
            A,B1,B2 = A[...,None],B1[...,None],B2[...,None]
            b = torch.stack([
                (1/12)*(s1*(-4*A + 4*B1 + 4*B2 + 5) + s2*(-4*A + 4*B1 + 4*B2 + 5) + 2*s3*(-2*A + 2*B1 + 2*B2 + 3))/A, 
                (1/12)*(-2*s1*(2*B1 + 1) - s2*(4*B1 + 3) - s3*(4*B1 + 3))/A, 
                (1/12)*(-s1*(4*B2 + 3) - 2*s2*(2*B2 + 1) - s3*(4*B2 + 3))/A
            ],-2).reshape(N,3,2)
    elif LU0 == 'lower':
        if LU1 =='upper':
            M = torch.stack([
                (1/6)*(6*A**2 - 12*A*B1 - 4*A + 6*B1**2 + 4*B1 + 1)/A**2, (A**2 - 1/3*A*(3*B1 + 3*B2 + 2) + B1*B2 + (1/3)*B1 + (1/3)*B2 + 1/12)/A**2, 
                (-A**2 + A*(2*B1 + B2 + 1) - B1**2 - B1*B2 - B1 - 1/3*B2 - 1/4)/A**2, (A**2 - 1/3*A*(3*B1 + 3*B2 + 2) + B1*B2 + (1/3)*B1 + (1/3)*B2 + 1/12)/A**2, 
                (1/6)*(6*A**2 - 12*A*B2 - 4*A + 6*B2**2 + 4*B2 + 1)/A**2, (-A**2 + A*(B1 + 2*B2 + 1) - B1*B2 - 1/3*B1 - B2**2 - B2 - 1/4)/A**2, 
                (-A**2 + A*(2*B1 + B2 + 1) - B1**2 - B1*B2 - B1 - 1/3*B2 - 1/4)/A**2, (-A**2 + A*(B1 + 2*B2 + 1) - B1*B2 - 1/3*B1 - B2**2 - B2 - 1/4)/A**2, 
                (1/6)*(6*A**2 - 4*A*(3*B1 + 3*B2 + 2) + 6*B1**2 + 12*B1*B2 + 8*B1 + 6*B2**2 + 8*B2 + 3)/A**2
            ],-1).reshape(N,3,3)
            A,B1,B2 = A[...,None],B1[...,None],B2[...,None]
            b = torch.stack([
                (1/12)*(s1*(-4*A + 4*B1 + 1) + 2*s2*(-2*A + 2*B1 + 1) + s3*(-4*A + 4*B1 + 1))/A, 
                (1/12)*(s1*(-4*A + 4*B2 + 1) + s2*(-4*A + 4*B2 + 1) + 2*s3*(-2*A + 2*B2 + 1))/A, 
                (1/12)*(-2*s1*(-2*A + 2*B1 + 2*B2 + 1) - s2*(-4*A + 4*B1 + 4*B2 + 3) - s3*(-4*A + 4*B1 + 4*B2 + 3))/A
            ],-2).reshape(N,3,2)
        elif LU1 == 'lower':
            M=torch.stack([
                (1/6)*(6*A**2 - 12*A*B1 - 8*A + 6*B1**2 + 8*B1 + 3)/A**2, (1/12)*(12*A**2 - 4*A*(3*B1 + 3*B2 + 4) + 12*B1*B2 + 8*B1 + 8*B2 + 5)/A**2, 
                (-A**2 + A*(2*B1 + B2 + 2) - B1**2 - B1*B2 - 2*B1 - 2/3*B2 - 11/12)/A**2, (1/12)*(12*A**2 - 4*A*(3*B1 + 3*B2 + 4) + 12*B1*B2 + 8*B1 + 8*B2 + 5)/A**2, 
                (1/6)*(6*A**2 - 12*A*B2 - 8*A + 6*B2**2 + 8*B2 + 3)/A**2, (-A**2 + A*(B1 + 2*B2 + 2) - B1*B2 - 2/3*B1 - B2**2 - 2*B2 - 11/12)/A**2, 
                (-A**2 + A*(2*B1 + B2 + 2) - B1**2 - B1*B2 - 2*B1 - 2/3*B2 - 11/12)/A**2, (-A**2 + A*(B1 + 2*B2 + 2) - B1*B2 - 2/3*B1 - B2**2 - 2*B2 - 11/12)/A**2, 
                (1/6)*(6*A**2 - 4*A*(3*B1 + 3*B2 + 4) + 6*B1**2 + 12*B1*B2 + 16*B1 + 6*B2**2 + 16*B2 + 11)/A**2
            ],-1).reshape(N,3,3)
            A,B1,B2 = A[...,None],B1[...,None],B2[...,None]
            b = torch.stack([
                (1/12)*(2*s1*(-2*A + 2*B1 + 1) + s2*(-4*A + 4*B1 + 3) + s3*(-4*A + 4*B1 + 3))/A, 
                (1/12)*(s1*(-4*A + 4*B2 + 3) + 2*s2*(-2*A + 2*B2 + 1) + s3*(-4*A + 4*B2 + 3))/A, 
                (1/12)*(-s1*(-4*A + 4*B1 + 4*B2 + 5) - s2*(-4*A + 4*B1 + 4*B2 + 5) - 2*s3*(-2*A + 2*B1 + 2*B2 + 3))/A
            ],-2).reshape(N,3,2)
    
    c = (1/12)*s1**2 + (1/12)*s1*s2 + (1/12)*s1*s3 + (1/12)*s2**2 + (1/12)*s2*s3 + (1/12)*s3**2
    return M,b,c






if __name__ == '__main__':
    
    parser = ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--normal',dest="normal",action='store_true')
    parser.set_defaults(normal=False)
    args = parser.parse_args()
    
    INPUT = args.input # path to normal map
    filename = INPUT.split('/')[-1].split('.')[0]
    OUTPUT = args.output # path to output folder
    read_normal = args.normal # is heigh field or normal map

    
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

    
    normal_patch = normal[...,:2]
    if normal_patch.shape[0]%1024 == 0: # 1025 vertex, 1024 cell
        normal_patch = torch.cat([normal_patch,normal_patch[:,0:1]],1)
        normal_patch = torch.cat([normal_patch,normal_patch[0:1]],0)

        
    # optimize cluster normal   
    H = normal_patch.shape[0]
    use_detJ = True
    normal_est = []
    eps = 1e-6
    scales = [2,4,8,16,32]
    '''
    0-1
    |/|
    2-3
    0,1,2
    2,1,3
    '''
    print("build cluster hierarchy")
    for scale in scales:
        A = torch.ones((H//scale)*(H//scale)).mul(scale)
        B1,B2 = torch.ones_like(A),torch.ones_like(A)
        MA = torch.zeros(len(A),4,4)
        Mb = torch.zeros(len(A),4,2)
        Mc = torch.zeros(len(A),2)
        for i in range(scale):
            for j in range(scale-i):
                s1 = normal_patch[i:-1:scale,j:-1:scale,:2].reshape(-1,2)
                s2 = normal_patch[i:-1:scale,j+1::scale,:2].reshape(-1,2)
                s3 = normal_patch[i+1::scale,j:-1:scale,:2].reshape(-1,2)
                MA_,Mb_,Mc_ = triangle2Matrix(s1,s2,s3,A,B1*j,B2*i,'upper','upper')
                if use_detJ:
                    detJ = triangle_area(s1,s2,s3).clamp_min(eps)
                else:
                    detJ = torch.ones_like(s1[...,0])
                MA[:,:3,:3] += MA_/detJ[:,None,None]
                Mb[:,:3] += Mb_/detJ[:,None,None]
                Mc += Mc_/detJ[:,None]
        for i in range(scale-1):
            for j in range(scale-i-1):
                s1 = normal_patch[i+1::scale,j:-1:scale,:2].reshape(-1,2)
                s2 = normal_patch[i:-1:scale,j+1::scale,:2].reshape(-1,2)
                s3 = normal_patch[i+1::scale,j+1::scale,:2].reshape(-1,2)
                MA_,Mb_,Mc_ = triangle2Matrix(s1,s2,s3,A,B1*j,B2*i,'upper','lower')
                if use_detJ:
                    detJ = triangle_area(s1,s2,s3).clamp_min(eps)
                else:
                    detJ = torch.ones_like(s1[...,0])
                MA[:,:3,:3] += MA_/detJ[:,None,None]
                Mb[:,:3] += Mb_/detJ[:,None,None]
                Mc += Mc_/detJ[:,None]

        for i in range(scale):
            for j in range(scale-i-1,scale):
                s1 = normal_patch[i+1::scale,j:-1:scale,:2].reshape(-1,2)
                s2 = normal_patch[i:-1:scale,j+1::scale,:2].reshape(-1,2)
                s3 = normal_patch[i+1::scale,j+1::scale,:2].reshape(-1,2)
                MA_,Mb_,Mc_ = triangle2Matrix(s1,s2,s3,A,B1*j,B2*i,'lower','lower')
                if use_detJ:
                    detJ = triangle_area(s1,s2,s3).clamp_min(eps)
                else:
                    detJ = torch.ones_like(s1[...,0])
                MA[:,1:,1:] += (MA_/detJ[:,None,None])[:,[1,0,2]][:,:,[1,0,2]]
                Mb[:,1:] += (Mb_/detJ[:,None,None])[:,[1,0,2]]
                Mc += Mc_/detJ[:,None]

        for i in range(1,scale):
            for j in range(scale-i,scale):
                s1 = normal_patch[i:-1:scale,j:-1:scale,:2].reshape(-1,2)
                s2 = normal_patch[i:-1:scale,j+1::scale,:2].reshape(-1,2)
                s3 = normal_patch[i+1::scale,j::scale,:2].reshape(-1,2)
                MA_,Mb_,Mc_ = triangle2Matrix(s1,s2,s3,A,B1*j,B2*i,'lower','upper')
                if use_detJ:
                    detJ = triangle_area(s1,s2,s3).clamp_min(eps)
                else:
                    detJ = torch.ones_like(s1[...,0])
                MA[:,1:,1:] += (MA_/detJ[:,None,None])[:,[1,0,2]][:,:,[1,0,2]]
                Mb[:,1:] += (Mb_/detJ[:,None,None])[:,[1,0,2]]
                Mc += Mc_/detJ[:,None]

        n_est = torch.linalg.lstsq(MA,-Mb)[0]
        residuals = (MA@n_est+Mb).pow(2).sum([-1,-2]).sqrt()*scale*scale
        normal_est.append(torch.cat([residuals.reshape(-1,1),n_est.reshape(-1,8)],1))
        
    # save cluster hierarchy
    os.makedirs(f'{OUTPUT}/{filename}/',exist_ok=True)
    torch.cat(normal_est,0).reshape(-1).numpy().tofile(f'{OUTPUT}/{filename}/tree')
        
        
    # save normal map
    normal[...,:2].numpy().astype('float32').tofile(f'{OUTPUT}/{filename}/normal')
        
        
    # save bounding box hierarchy
    print("build bounding box hierarchy")
    sigma_max = 128
    pad = (normal.shape[0]%1024 == 0)
    normals = normal[...,:2].float().clone()
    if pad:
        normals = torch.cat([normals,normals[:,:1]],1)
        normals = torch.cat([normals,normals[:1]],0)
    bound_min = torch.min(
                    torch.min(normals[:-1,:-1],normals[:-1,1:]),
                    torch.min(normals[1:,:-1],normals[1:,1:]))
    bound_max = torch.max(
                    torch.max(normals[:-1,:-1],normals[:-1,1:]),
                    torch.max(normals[1:,:-1],normals[1:,1:]))
    level = int(np.log2(sigma_max))+1
    bounds = [torch.cat([bound_min,bound_max],-1).reshape(-1)]
    for l in range(1,level+1):
        bound_min = -NF.max_pool2d(-bound_min.permute(2,0,1)[None],2,2)[0].permute(1,2,0)
        bound_max = NF.max_pool2d(bound_max.permute(2,0,1)[None],2,2)[0].permute(1,2,0)
        bounds.append(torch.cat([bound_min,bound_max],-1).reshape(-1))

    bounds.insert(0,torch.tensor([level]))
    bounds = torch.cat(bounds,0)
    bounds.numpy().astype('float32').tofile(f'{OUTPUT}/{filename}/bound')