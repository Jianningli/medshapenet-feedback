#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset

class DataSet_GS(Dataset):
    """
    Load in 3D medical image, treate image as a stack of 2D images with given dimension
    """
    def __init__(self, input_image, transform = None):        
        super(DataSet_GS, self).__init__()
        self.input_image = input_image
        
        
    def __len__(self):
        return np.shape(self.input_image)[0]
    
    def __getitem__(self, idx):
        im_sample = self.input_image[idx, :,...]
        sample = {'input_image': im_sample}
        return sample



