from scipy.io import loadmat,savemat
import numpy as np
import torch

class PCAEncoder:
    def __init__(self, model_pth, l=128, crop_l=79, dim=42,average=True):
        self.model_pth=model_pth
        self.l=l
        self.dim=dim
        self.crop_l=crop_l
        self.average=average

    def initialize(self,device):
        data=loadmat(self.model_pth)
        self.matrix=torch.from_numpy(data["matrix"]).float().to(device) # [crop_l*crop_l,dim]
        self.mean=torch.from_numpy(data["mean"]).float().to(device) # [1, crop_l*crop_l]

    def encode(self,batch_kernel):
        # batch_kernel is [B,1,crop_l,crop_l], return [B,dim]
        B, _, H, W = batch_kernel.size() #[B, l, l]
        batch_kernel=batch_kernel.view((B, H * W))
        if self.average:
            batch_kernel=batch_kernel-self.mean
        reduced_kernel=torch.matmul(batch_kernel, self.matrix) # [B,dim]
        
        return reduced_kernel
    
    def decode(self,reduced_kernel):
        # reduced_kernel is [B,dim], return [B,crop_l,crop_l]
        kernel = torch.matmul(reduced_kernel, self.matrix.T)  # (B, H*W)
        if self.average:
            kernel = kernel + self.mean  # (B, H*W)

        kernel=kernel.view((-1, self.crop_l,self.crop_l)) # (B, H, W)
        return kernel
