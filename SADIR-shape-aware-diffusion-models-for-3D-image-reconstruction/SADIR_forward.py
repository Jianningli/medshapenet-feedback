import torch, os, glob, gc
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['NEURITE_BACKEND'] = 'pytorch'
from torch.autograd import Variable
from scipy import ndimage
import enum
import torchvision
import math
import nibabel as nib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from voxelmorph import voxelmorph
import numpy as np
import time
import lagomorph as lm
from lagomorph import adjrep 
from lagomorph import deform 
import SimpleITK as sitk
import random


IMAGE_SIZE=64
device = torch.device('cuda')

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    #print("extract tensor", arr.shape, timesteps)
    res = torch.from_numpy(arr)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def get_diffused_image(x_start, Time=1000):
    
    betas = get_named_beta_schedule("linear", Time)
    betas = np.array(betas, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
    t=Time-1
    noise = torch.randn_like(x_start)

    assert noise.shape == x_start.shape
    x_t= (
            _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
    )
    return x_t


def EPDiff_step(metric, m0, dt, phiinv, mommask=None):
    m = adjrep.Ad_star(phiinv, m0)
    if mommask is not None:
        m = m * mommask
    v = metric.sharp(m)
    return deform.compose_disp_vel(phiinv, v, dt=-dt), m, v

def lagomorph_expmap_shooting(metric, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
    d = len(m0.shape)-2

    if phiinv is None:
        phiinv = torch.zeros_like(m0)

    if checkpoints is None or not checkpoints:
        dt = T/num_steps
        for i in range(num_steps):
            phiinv, m, v = EPDiff_step(metric, m0, dt, phiinv, mommask=mommask)
            
    return phiinv

def get_deformed_image(m0, src):
    device = torch.device('cuda')
    imagesize = IMAGE_SIZE
    num_steps = 10
    alpha=0.5; gamma = 1.0; alpha=3
    fluid_params = [alpha, 0, gamma]
    metric = lm.FluidMetric(fluid_params)
    src = src.permute(0,1,4,3,2)
    phiinv = lagomorph_expmap_shooting(metric, m0, num_steps=num_steps)
    Sdef = lm.interp(src, phiinv)

    Sdef = Sdef.permute(0,4,3,2,1)[0]
    phiinv = phiinv.permute(0,4,3,2,1)[0]
    return Sdef
