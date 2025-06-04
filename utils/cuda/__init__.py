import torch
from torch.utils.cpp_extension import load
from pathlib import Path



_ext_src_root = Path(__file__).parent / 'src'
exts = ['.cpp', '.cu']
_ext_src_files = [
    str(f) for f in _ext_src_root.iterdir() 
    if any([f.name.endswith(ext) for ext in exts])
]

_ext = load(name='custom_cuda_ext', 
            sources=_ext_src_files)

def project_triangle(p0,p1,p2,c):
    #ret = torch.zeros(len(p0),3,2,3,device=p0.device)
    ret = torch.zeros(len(p0),device=p0.device)
    _ext.project_triangle(p0.contiguous(),p1.contiguous(),p2.contiguous(),
                          c, ret.contiguous())
    return ret


def vis_clip(p0,p1,p2,c):
    ret = torch.zeros(len(p0),3,2,3,device=p0.device)
    _ext.vis_clip(p0.contiguous(),p1.contiguous(),p2.contiguous(),
                          c, ret.contiguous())
    return ret