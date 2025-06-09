from scipy.io import loadmat,savemat
import numpy as np
import torch

class PCAEncoder:
    def __init__(self, model_pth, l=128, crop_l=79, b=42,average=False):
        self.model_pth=model_pth
        self.l=l
        self.b=b
        self.crop_l=crop_l
        self.average=average

    def initialize(self,device):
        data=loadmat(self.model_pth)
        self.matrix=torch.from_numpy(data["matrix"]).float().to(device)
        self.mean=torch.from_numpy(data["mean"]).float().to(device) # [1, crop_l*crop_l]

    def encode(self,batch_kernel):
        # kernel is [M,1,crop_l,crop_l]，返回[M,b]
        B, _, H, W = batch_kernel.size() #[B, l, l]
        batch_kernel=batch_kernel.view((B, 1, H * W))
        if self.average: #需要让均值在0附近
            batch_kernel=batch_kernel-self.mean
        reduced_kernel=torch.bmm(batch_kernel, self.matrix.expand((B, ) + self.matrix.shape)).view((B, -1)) # [M,b]
        reduced_kernel = torch.nn.functional.normalize(reduced_kernel, p=2, dim=1) # 每行变为求和为1的向量，不然数值太小不容易计算误差
        return reduced_kernel
    
    def decode(self,reduced_kernel):
        # reduced_kernel is [M,b], 返回[M,crop_l,crop_l]
        B,b=reduced_kernel.size()
        mat=self.matrix.transpose(0,1)
        kernel=torch.bmm(reduced_kernel.view(B,1,b), mat.expand((B, ) + mat.shape)) #[B,1,crop_l*crop_l]
        if self.average:
            kernel=kernel+self.mean
        kernel=kernel.view((B, self.crop_l,self.crop_l))
        return kernel
