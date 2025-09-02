import os
from matplotlib.image import imread
from torch.utils.data import Dataset
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_nl_means
from glob import glob
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
    obj_files=sorted(obj_files) # sorted to prevent repetition
    return obj_files

class TiltDataset(Dataset):
    def __init__(self, data_dir,mode='train',ratio=0.8,imgsize=128,crop_l=79):
        
        self.data_dir = os.path.join(data_dir, "data/")
        self.calib_dir =os.path.join(data_dir, "calib/")

        self.files = find_files(self.data_dir,'mat')

        self.h=imgsize
        self.w=imgsize
        self.crop_l=crop_l
        self.start_index=int(imgsize/2-(crop_l-1)/2)
        
        if mode=='train':
            start=0
            end=int(ratio*len(self.files))
        elif mode=='val':
            start=int(ratio*len(self.files))+1
            end=len(self.files)

        self.files=self.files[start:end]

    def  __len__(self):
        return len(self.files)
    
    def transform(self, img, xoffset, yoffset, degree_angle):
        w, h = img.shape

        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree_angle, 1.0)
        # img = cv2.warpAffine(img, rotation_matrix, (w, h), borderMode=cv2.INTER_LINEAR)  # Fill using interpolation
        img = cv2.warpAffine(img, rotation_matrix, (w, h))  # Fill empty areas with zeros
        img = cv2.resize(img, (w, h))

        M = np.float32([[1, 0, xoffset], [0, 1, yoffset]])  # Translation matrix: +100 in x direction, +50 in y direction
        # img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.INTER_LINEAR)  # Fill using interpolation
        img = cv2.warpAffine(img, M, (w, h))  # Fill empty areas with zeros
        return img
    
    def normalize(self,img):
        minvalue=np.min(img)
        maxvalue=np.max(img)
        return (img-minvalue)/(maxvalue-minvalue)
        
    def __getitem__(self, i):
        try:
            ## Random rotation
            rotate=False
            if random.random()>0.8:
                rotate=False

            # Read images
            img=loadmat(self.files[i])['img']
            img=np.squeeze(img)
            
            if img.shape[0]>self.h:
                startx=np.random.randint(0, img.shape[0]-self.h)  ## Random crop
                starty=np.random.randint(0, img.shape[1]-self.w)
                img=img[startx:startx+self.h,starty:starty+self.w]
            if rotate:
                img=img.T
            img=self.normalize(img)
            img=torch.from_numpy(img).unsqueeze(0)

            temp=self.files[i].replace("\\","/").split("/")

            # Read kernels
            psffile=self.calib_dir+temp[-1]
            psfimg=loadmat(psffile)['img']
            psfimg=np.squeeze(psfimg)
            psfimg = cv2.resize(psfimg, (self.h,self.w))
            psfimg=psfimg[self.start_index:self.start_index+self.crop_l,self.start_index:self.start_index+self.crop_l]
            psfimg=psfimg/np.sum(psfimg)
            psfimg=torch.from_numpy(psfimg).unsqueeze(0)
            
            # Read gt images
            gtfile = re.sub(r'alpha\d+', 'alpha0', self.files[i])
            gtimg=loadmat(gtfile)['img']
            gtimg=np.squeeze(gtimg)
            if gtimg.shape[0]>self.h:
                gtimg=gtimg[startx:startx+self.h,starty:starty+self.w]
            if rotate:
                gtimg=gtimg.T
            gtimg=self.normalize(gtimg)
            gtimg=torch.from_numpy(gtimg).unsqueeze(0)

            # Return torch tensor
            dic = {'gt': gtimg,  'img':img, 'psf': psfimg}
            return dic
        
        except Exception as e:
            print(self.files[i])
            print(e)
            exit()
            return {'gt': torch.zeros(1,self.h,self.w),  'img':torch.zeros(1,self.h,self.w), 'psf': torch.zeros(1,self.crop_l,self.crop_l)}

class ThermalNLOSDataset(Dataset):
    def __init__(self, data_dir="ThermalNLOS/",name='augment',mode=None,ratio=0.9,imgsize=128,crop_l=79):
        # `name` can be "base","augment","test"
        # `mode` can be "train" or "val"
        
        self.data_dir = data_dir+name+"/" # the path of training data
        
        self.files = find_files(self.data_dir,'mat')

        self.h=imgsize
        self.w=imgsize
        self.crop_l=crop_l
        self.start_index=int(imgsize/2-(crop_l-1)/2)
        
        if mode=='train':
            start=0
            end=int(ratio*len(self.files))
        elif mode=='val':
            start=int(ratio*len(self.files))+1
            end=len(self.files)
        else:
            start=0
            end=len(self.files)

        self.files=self.files[start:end]

    def  __len__(self):
        return len(self.files)
    
    def normalize(self,img):
        minvalue=np.min(img)
        maxvalue=np.max(img)
        return (img-minvalue)/(maxvalue-minvalue)
    
    def random_noise(self,shape,maxvalue):
        noise = np.random.uniform(0,maxvalue,size=shape)
        return noise.astype(np.float32)
    
    def get_gaussian():
        return
        
    def __getitem__(self, i):
        try:
            # measurements
            data=loadmat(self.files[i])
            img=np.squeeze(data["img"])
            img=self.normalize(img)
            img=img+self.random_noise(img.shape,0.2)
            img=self.normalize(img)
            img=torch.from_numpy(img).unsqueeze(0)

            # read psf image
            psfimg=data['psf']
            psfimg=np.squeeze(psfimg)
            psfimg=psfimg[self.start_index:self.start_index+self.crop_l,self.start_index:self.start_index+self.crop_l]
            psfimg=psfimg/np.sum(psfimg)
            psfimg=torch.from_numpy(psfimg).unsqueeze(0)
            
            # read gt image
            gtimg=data['gt']
            gtimg=np.squeeze(gtimg)
            gtimg=self.normalize(gtimg)
            gtimg=torch.from_numpy(gtimg).unsqueeze(0)

            dic = {'gt': gtimg,  'img':img, 'psf': psfimg}
            return dic
        
        except Exception as e:
            print(self.files[i])
            print(e)
            exit()
            return {'gt': torch.zeros(1,self.h,self.w),  'img':torch.zeros(1,self.h,self.w), 'psf': torch.zeros(1,self.crop_l,self.crop_l)}
        

class BlendDataset(Dataset):
    def __init__(self, data_dir,mode='train',ratio=0.8,imgsize=128,crop_l=79):
        self.data_dir = data_dir
        if type(data_dir)==type([]):
            self.files = []
            for dir in data_dir:
                temp=find_files(dir,'mat')
                self.files=self.files+temp
        else:
            self.files = find_files(data_dir,'mat')

        self.h=imgsize
        self.w=imgsize
        self.crop_l=crop_l
        self.start_index=int(imgsize/2-(crop_l-1)/2)
        
        if mode=='train':
            start=0
            end=int(ratio*len(self.files))
        elif mode=='val':
            start=int(ratio*len(self.files))+1
            end=len(self.files)

        self.files=self.files[start:end]

    def  __len__(self):
        return len(self.files)
    
    def normalize(self,img):
        minvalue=np.min(img)
        maxvalue=np.max(img)
        return (img-minvalue)/(maxvalue-minvalue)
    
    def random_noise(self,shape,maxvalue):
        noise = np.random.uniform(0,maxvalue,size=shape)
        return noise.astype(np.float32)
    
    def get_gaussian():
        return
        
    def __getitem__(self, i):
        try:
            # Read measurements
            data=loadmat(self.files[i])
            img=np.squeeze(data["img"])
            img=self.normalize(img)
            img=img+self.random_noise(img.shape,0.2)
            img=self.normalize(img)
            img=torch.from_numpy(img).unsqueeze(0)

            # Read kernels
            psfimg=data['psf']
            psfimg=np.squeeze(psfimg)
            psfimg=psfimg[self.start_index:self.start_index+self.crop_l,self.start_index:self.start_index+self.crop_l]
            psfimg=psfimg/np.sum(psfimg)
            psfimg=torch.from_numpy(psfimg).unsqueeze(0)
            
            # Read gt images
            gtimg=data['gt']
            gtimg=np.squeeze(gtimg)
            gtimg=self.normalize(gtimg)
            gtimg=torch.from_numpy(gtimg).unsqueeze(0)

            # Return torch tensor
            dic = {'gt': gtimg,  'img':img, 'psf': psfimg}
            return dic
        
        except Exception as e:
            print(self.files[i])
            print(e)
            exit()
            return {'gt': torch.zeros(1,self.h,self.w),  'img':torch.zeros(1,self.h,self.w), 'psf': torch.zeros(1,self.crop_l,self.crop_l)}
        
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
            # Read measurements
            image=self._readTiff(self.files[i])
            if self.back:
                image=image-self.backimg
            
            # Non-local mean filter
            # image = denoise_nl_means(image/np.max(image),h=1.0,fast_mode=True,patch_size=5,patch_distance=6,channel_axis=None)

            # Gaussian filter
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

class MatDataset(Dataset):
    def __init__(self, data_dir,imgsize=128, filter=False):
        
        self.data_dir = data_dir
        
        self.files = find_files(data_dir,'mat')

        self.imgsize=imgsize
        self.filter=filter

    def  __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            # load measurements
            image=loadmat(self.files[i])['img']
            image=np.squeeze(image).astype(np.float32)

            if self.filter:
                # Gaussian filter
                # image=(image-np.min(image))/(np.max(image)-np.min(image))
                # image = cv2.GaussianBlur(image, (11,11), sigmaX=2,sigmaY=2)
                
                # Non-local mean filter
                image = denoise_nl_means(image/np.max(image),h=1.0,fast_mode=True,patch_size=5,patch_distance=4,channel_axis=None)

            image = cv2.resize(image[:,image.shape[1]-image.shape[0]:], (self.imgsize,self.imgsize))
            # image = cv2.resize(image[:,:image.shape[0]], (self.imgsize,self.imgsize))
            image=(image-np.min(image))/(np.max(image)-np.min(image))

            dic = {'img':torch.from_numpy(image).unsqueeze(0)}
            return dic
        
        except Exception as e:
            print(self.files[i])
            print(e)
    
if __name__=='__main__':
    # dataset=DeconvDataset("/home/yrl/Infrared/car_data")
    dataset=TiltDataset("/data/yrl/Infrared/alpha_data_tilt","val")
    print(dataset.calib_dir)
    print(dataset.data_dir)
    print(len(dataset))
    print(dataset[0]["img"].shape)
    print(dataset[0]["psf"].shape)