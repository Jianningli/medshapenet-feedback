from __future__ import division, print_function
from typing import Dict, SupportsRound, Tuple, Any
from os import PathLike
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch,gc
from torch.autograd import grad
from torch.autograd import Variable
import torch.fft ############### Pytorch >= 1.8.0
import torch.nn.functional as F
import SimpleITK as sitk
import os, glob
import json
import subprocess
import nibabel as nib
import sys
from PIL import Image
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from easydict import EasyDict as edict
from sklearn.metrics import accuracy_score
import random, shutil, os
import yaml
from DataSet import DataSet_GS
import lagomorph
from networks import VxmDense  
from losses import MSE, Grad


IMAGE_SIZE = 64
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
    

def loss_Reg(y_pred):
        dz = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dy = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        dz = dz * dz
        dy = dy * dy
        dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz) 
        grad = d / 3.0
        return grad
    



def define_data():
    datapath = './train_64/'
    len_ = len(os.listdir(datapath))
    # data = json.load(open(readfilename, 'r'))
    outputs = []
    keyword = 'train'
    ave_scan = np.zeros((xDim,yDim,zDim))
    source_scan = np.zeros((xDim,yDim,zDim))
    for f_ in os.listdir(datapath):
        if not f_.startswith('.'):
            filename_src = datapath + f_
            itkimage_src = sitk.ReadImage(filename_src)
            source_scan = sitk.GetArrayFromImage(itkimage_src)
            source_scan = resize_volume(source_scan, 128, 64)
            ave_scan = ave_scan + source_scan
            outputs.append(source_scan)

    ave_scan = torch.FloatTensor((ave_scan/len_).reshape(1,xDim,yDim,zDim))
    train = torch.FloatTensor(outputs)

    training = DataSet_GS(input_image = train)
    return training




def main_atlas_train(atlas_exists):
    net = []
    for i in range(3):
        temp = VxmDense(inshape = (xDim,yDim,zDim),
                     nb_unet_features= [[16, 16, 32],[32, 32, 16, 16]], ### changed the dimension
                     nb_unet_conv_per_level=1,
                     int_steps=7,
                     int_downsize=2,
                     src_feats=1,
                     trg_feats=1,
                     unet_half_res= True)
        net.append(temp)
    net = net[0].to(dev)

    trainloader = torch.utils.data.DataLoader(define_data(), batch_size = para.solver.batch_size, shuffle=True, num_workers=1)

    running_loss = 0 
    running_loss_val = 0
    template_loss = 0
    printfreq = 1
    sigma = 0.02
    repara_trick = 0.0
    loss_array = torch.FloatTensor(para.solver.epochs,1).fill_(0)
    loss_array_val = torch.FloatTensor(para.solver.epochs,1).fill_(0)

    atlas = torch.cuda.FloatTensor(1, 1, xDim, yDim, zDim).fill_(0).contiguous()
    atlas.requires_grad=True
    ave_scan = ave_scan.to(dev)
    ave_scan.requires_grad=True

    if(para.model.loss == 'L2'):
        criterion = nn.MSELoss()
    elif (para.model.loss == 'L1'):
        criterion = nn.L1Loss()
    if(para.model.optimizer == 'Adam'):
        optimizer = optim.Adam(net.parameters(), lr= para.solver.lr)
    elif (para.model.optimizer == 'SGD'):
        optimizer = optim.SGD(net.parameters(), lr= para.solver.lr, momentum=0.9)
    if (para.model.scheduler == 'CosAn'):
        scheduler = CosineAnnealingLR(optimizer, T_max=len(trainloader), eta_min=0)

    criterion_SIC = nn.CrossEntropyLoss() 
    optimizer_template = optim.Adam(net.parameters(), lr= para.solver.lr)
    scheduler_template = CosineAnnealingLR(optimizer_template, T_max=len(trainloader), eta_min=0)

    loss_plot = []
    if atlas_exists:
        atlas= torch.from_numpy(np.array(nib.load('final_atlas.nii').get_fdata())).unsqueeze(0).float().cuda()
    for epoch in range(200):
        total= 0; 
        count = 0
        total_val = 0; 
        total_template = 0; 
        net.train()

        for j, atlas_data in enumerate(trainloader):
            torch.cuda.empty_cache()
            inputs = atlas_data['input_image'].to(dev)
            b, c, w, h, l = inputs.shape
            optimizer.zero_grad()

            atlas = ave_scan

            source_b = torch.cat(b*[atlas]).reshape(b , c, w , h, l)

            pred = net(source_b, inputs, registration = True)     
            loss = criterion(pred[0], inputs) 
            # loss2 = (pred[1])
            loss2 = loss_Reg(pred[1])
            # v0_pred= pred[1]
            # v0_pred.requires_grad=True
            loss_total = loss + 0.1*loss2

            loss_total.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_total.item()
            total += running_loss
            running_loss = 0.0
            with torch.no_grad():
                atlas.data = atlas.data - 0.01*atlas.grad
            count += 1

        print('epoch: ', epoch, 'loss: ', total/count)
        loss_plot.append(total/count)

    save_path = './final_atlas.nii'
    sitk.WriteImage(sitk.GetImageFromArray(atlas.squeeze().detach().cpu().numpy(), isVector=False), save_path,False)