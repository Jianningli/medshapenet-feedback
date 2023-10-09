from __future__ import division, print_function
from typing import Dict, SupportsRound, Tuple, Any
from os import PathLike
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch,gc
from torch.autograd import grad
from torch.autograd import Variable
import torch.fft ############### Pytorch >= 1.8.0
import torch.nn.functional as F
import SimpleITK as sitk
import os, glob
import json, itk
import subprocess
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
import random
import yaml
from DataSet import DataSet_GS


################ Device Seting #######################
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

################ Parameter Loading #######################
def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None
para = read_yaml('./parameters.yml')
xDim = para.data.x 
yDim = para.data.y
zDim = para.data.z

##################Data Loading##########################
readfilename = './2DBrainfull_normalized/data' + '.json'
datapath = './2DBrainfull_normalized/'
data = json.load(open(readfilename, 'r'))

################## Training Data Loading##########################
outputs = []
keyword = 'train'
ave_scan = np.zeros((1,xDim,yDim))
source_scan = np.zeros((1,xDim,yDim))
for i in range (0,len(data[keyword])):
    filename_src = datapath + data[keyword][i]['image']
    itkimage_src = sitk.ReadImage(filename_src)
    source_scan = sitk.GetArrayFromImage(itkimage_src)
    ave_scan = ave_scan + source_scan
    outputs.append(source_scan)
ave_scan = torch.FloatTensor((ave_scan/len(data[keyword])).reshape(1,1,xDim,yDim))
rand_scan= torch.FloatTensor(source_scan.reshape(1,1,xDim,yDim))
train = torch.FloatTensor(outputs)
train = train.reshape(len(data[keyword]),1,xDim,yDim)
print (train.shape)


class_label = []
for i in range (0,len(data[keyword])):
    templabel = int(data[keyword][i]['label'])
    class_label.append(templabel)

train_label = torch.tensor(class_label, dtype=torch.long)
print (train_label.shape)
training = DataSet_GS (input_image = train, groundtruth = train_label )

################## Validation Data Loading##########################
outputs = []
keyword = 'val'
# outputs = np.array(outputs)
ave_scan = np.zeros((1,xDim,yDim))
source_scan = np.zeros((1,xDim,yDim))
for i in range (0,len(data[keyword])):
    filename_src = datapath + data[keyword][i]['image']
    itkimage_src = sitk.ReadImage(filename_src)
    source_scan = sitk.GetArrayFromImage(itkimage_src)
    ave_scan = ave_scan + source_scan
    outputs.append(source_scan)
ave_scan = torch.FloatTensor((ave_scan/len(data[keyword])).reshape(1,1,xDim,yDim))
rand_scan= torch.FloatTensor(source_scan.reshape(1,1,xDim,yDim))
val = torch.FloatTensor(outputs)
val= val.reshape (len(data[keyword]),1,xDim,yDim)
print (val.shape)

class_label = []
for i in range (0,len(data[keyword])):
    templabel = int(data[keyword][i]['label'])
    class_label.append(templabel)

val_label = torch.tensor(class_label, dtype=torch.long)
print (val_label.shape)
validation = DataSet_GS (input_image = val, groundtruth = val_label )

################# Network Setting########################
from classifiers import simpleeCNN
from networks import VxmDense  
from losses import MSE, Grad
net = []
for i in range(3):
    temp = VxmDense(inshape = (xDim,yDim),
				 nb_unet_features= [[16, 32],[ 32, 32, 16, 16]],
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res= True)
    net.append(temp)
net = net[0].to(dev)
# print (net)

net_SIC = simpleeCNN()
net_SIC = net_SIC.to(dev)
trainloader = torch.utils.data.DataLoader(training, batch_size = para.solver.batch_size, shuffle=True, num_workers=1)
valloader = torch.utils.data.DataLoader(validation, batch_size = para.solver.batch_size, shuffle=True, num_workers=1)

running_loss = 0 
running_loss_val = 0
template_loss = 0
printfreq = 1
sigma = 0.02
repara_trick = 0.0
loss_array = torch.FloatTensor(para.solver.epochs,1).fill_(0)
loss_array_val = torch.FloatTensor(para.solver.epochs,1).fill_(0)
# # loss_array = loss_array.to(device)

gradv_batch = torch.cuda.FloatTensor(para.solver.batch_size, 3, xDim, yDim).fill_(0).contiguous()
defIm_batch = torch.cuda.FloatTensor(para.solver.batch_size, 1, xDim, yDim).fill_(0).contiguous()
temp = torch.cuda.FloatTensor(para.solver.batch_size, 3, xDim, yDim).fill_(0).contiguous()
transformations = torch.cuda.FloatTensor(para.solver.batch_size, 3, xDim, yDim).fill_(0).contiguous() 
atlas = torch.cuda.FloatTensor(1, 1, xDim, yDim).fill_(0).contiguous()
atlas.requires_grad=True
ave_scan = ave_scan.to(dev)
rand_scan = rand_scan.to(dev)
ave_scan.requires_grad=True

# rand_scan = rand_scan.to(dev)
# atlas = atlas.mean(dim=0).view(1,1,128,128)

gradv_batch_val = torch.cuda.FloatTensor(1, 3, xDim, yDim).fill_(0).contiguous()
defIm_batch_val = torch.cuda.FloatTensor(1, 1, xDim, yDim).fill_(0).contiguous() 
temp_val = torch.cuda.FloatTensor(1, 3, xDim, yDim).fill_(0).contiguous()
# temp = temp.contiguous()
deform_size = [1, xDim, yDim]
params = list(net.parameters()) + list(net_SIC.parameters())
if(para.model.loss == 'L2'):
    criterion = nn.MSELoss()
elif (para.model.loss == 'L1'):
    criterion = nn.L1Loss()
if(para.model.optimizer == 'Adam'):
    optimizer = optim.Adam(net.parameters(), lr= para.solver.lr)
elif (para.model.optimizer == 'SGD'):
    optimizer = optim.SGD(net.parameters(), lr= para.solver.lr, momentum=0.9)
if (para.model.scheduler == 'CosAn'):
    scheduler = CosineAnnealingLR(optimizer, T_max=len(valloader), eta_min=0)
criterion_SIC = nn.CrossEntropyLoss() 
optimizer_SIC = optim.Adam(net_SIC.parameters(), lr= para.solver.lr)
opt = optim.Adam(params, lr= para.solver.lr)
optimizer_template = optim.Adam(net.parameters(), lr= para.solver.lr)
scheduler_template = CosineAnnealingLR(optimizer_template, T_max=len(valloader), eta_min=0)
temp = torch.cuda.FloatTensor(para.solver.batch_size, 32, 14, 14).fill_(0).contiguous()
# ##################Training###################################
def loss_Reg(y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        dy = dy * dy
        dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy) 
        grad = d / 2.0
        return grad
for epoch in range(para.solver.epochs):
    total= 0; 
    total_val = 0; 
    total_template = 0; 
    net.train()
    net_SIC.train()
    print('epoch:', epoch)
    for j, atlas_data in enumerate(trainloader):
        inputs = atlas_data['input_image'].to(dev)
        batch_labels = atlas_data['gt'].to(dev)
        b, c, w, h = inputs.shape
        opt.zero_grad()
        if(epoch <= para.solver.pre_train ):
            randidx = random.randint(0, 299)
            atlas= train[randidx,:,:,:].to(dev)
        else:
            atlas = ave_scan
        ########## Joint Traning for Geo-SIC ##########
        source_b = torch.cat(b*[atlas]).reshape(b , c, w , h)
        pred = net(source_b, inputs, registration = True)     
        loss = criterion(pred[0], inputs) 
        loss2 = loss_Reg(pred[1])
        loss = loss + 0.1*loss2
        pred = net_SIC(inputs, pred[2])
        loss_SIC = criterion_SIC(pred, batch_labels)
        #Only train SIC loss when Geo network is pre-trained
        if (epoch > para.solver.pre_train ):
            loss_total = .5*loss_SIC + loss
        else: 
             loss_total = loss
        loss_total.backward(retain_graph=True)
        opt.step()
        running_loss += loss_total.item()
        # print('[%d, %5d] loss: %.3f' %
        #     (epoch + 1, i + 1, running_loss ))
        total += running_loss
        running_loss = 0.0
        # # print (atlas.grad)
        if (epoch > para.solver.pre_train  ):
            with torch.no_grad():
                atlas.data = atlas.data - 0.01*atlas.grad
        ########################## Checking Atlas ############################
        # # for b_id in range (0, 5):
        # # deform = pred[0][0,:,:,:].reshape(xDim, yDim).detach().cpu().numpy()
        # # im= sitk.GetImageFromArray(deform, isVector=False)
        # # save_path = './checkdef/atlas' + str(epoch) + '.mhd'
        # # sitk.WriteImage(sitk.GetImageFromArray(deform, isVector=False), save_path,False)

        # atl = atlas.reshape(xDim, yDim).detach().cpu().numpy()
        # im= sitk.GetImageFromArray(atl, isVector=False)
        # save_path = './check_atlas_Geo_SIC/atlas' + str(epoch) + '.mhd'
        # sitk.WriteImage(sitk.GetImageFromArray(atl, isVector=False), save_path,False)
    # Validation
    acc_ave = 0
    for t, val_data in enumerate(valloader):
        inputs = val_data['input_image'].to(dev)
        batch_labels = val_data['gt'].to(dev)
        b, c, w, h = inputs.shape
        pred = net(source_b, inputs, registration = True)  
        pred = net_SIC(inputs, pred[2])
        pred_labels = torch.cuda.FloatTensor(b).fill_(0).contiguous()
        loss_SIC = criterion_SIC(pred, batch_labels)
        #Only train SIC loss when Geo network is pre-trained
        loss_total_val = loss_SIC
        running_loss_val += loss_total_val.item()
        total_val += running_loss_val
        running_loss_val = 0.0
        for h in range (0, b):
            if (pred[h,0]>pred[h,1]):
                pred_labels[h] = 0
            else:
                pred_labels[h] = 1
        # print (batch_labels.detach().cpu().numpy(), pred_labels.detach().cpu().numpy())
        acc= accuracy_score(batch_labels.detach().cpu().numpy(), pred_labels.detach().cpu().numpy())
        acc_ave += acc 
    loss_array_val[epoch] = acc_ave/(len(val)/para.solver.batch_size)
    print (loss_array_val[epoch])
    print ('total training loss:', total)
    print ('total validation loss:', total_val)
np.save ('./accuracy_no',loss_array_val.detach().cpu().numpy())





       
    
 
        


