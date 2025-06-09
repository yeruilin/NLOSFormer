import os
from matplotlib.image import imread
from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import re
import cv2
import random

import warnings

warnings.filterwarnings('error')

def find_files(directory,ext):
    obj_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                obj_files.append(os.path.join(root, file))
    obj_files=sorted(obj_files)
    return obj_files
        
class RealDataset(Dataset):
    def __init__(self,data_dir,imgsize=128,back=True):
        self.imgsize=imgsize
        self.back=back
        if back:
            self.backimg=self._readTiff(os.path.join(data_dir, "back.tif"))
            self.files=find_files(os.path.join(data_dir, "img"),"tif")
        else:
            self.files=find_files(data_dir,"tif")
    
    def  __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        try:
            # measurements
            image=self._readTiff(self.files[i])
            if self.back:
                image=image-self.backimg
            # image = cv2.GaussianBlur(image, (11,11), sigmaX=3,sigmaY=3)
            # image = cv2.GaussianBlur(image, (21,21), sigmaX=5,sigmaY=5)
            image = cv2.resize(image[:,image.shape[1]-image.shape[0]:], (self.imgsize,self.imgsize))
            
            image=(image-np.min(image))/(np.max(image)-np.min(image))

            dic = {'img':torch.from_numpy(image).unsqueeze(0)}
            return dic
        
        except Exception as e:
            print(self.files[i])
            print(e)
            return {'img':torch.zeros(1,self.imgsize,self.imgsize)}
            
    
    def _readTiff(self,path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image=np.squeeze(image).astype(np.float32)
        
        return image

class SingleImageDataset(Dataset):
    def __init__(self, data_dir,imgsize=128):
        
        self.data_dir = data_dir
        
        self.files = find_files(data_dir,'mat')

        self.imgsize=imgsize

    def  __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            image=loadmat(self.files[i])['img']
            image=np.squeeze(image).astype(np.float32)
            # image=image+np.random.rand(image.shape[0],image.shape[1]).astype(np.float32)*0.5
            # image = cv2.GaussianBlur(image, (11,11), sigmaX=2,sigmaY=2)
            image = cv2.resize(image[:,image.shape[1]-image.shape[0]:], (self.imgsize,self.imgsize))
            image=(image-np.min(image))/(np.max(image)-np.min(image))

            dic = {'img':torch.from_numpy(image).unsqueeze(0)}
            return dic
        
        except Exception as e:
            print(self.files[i])
            print(e)