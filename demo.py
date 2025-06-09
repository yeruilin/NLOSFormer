# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
from NLOSFormer import NLOSFormer
from PCAEncoder import PCAEncoder
from dataset import *
from Inferer import Inferer

# parameters
img_size=128
crop_l=79
pca_path="data/pca_matrix.mat"

# load model
model=NLOSFormer(input_channel=1,hidden_channel=64,psf_dim=42)
model_path="pth/model-03302340.pt"

pca_encoder=PCAEncoder(pca_path,l=img_size, crop_l=79, b=42,average=True)

# inference
simudataset=RealDataset("data/squat/",img_size,False)

inferer = Inferer(
    simudataset,
    model,
    pca_encoder,
    model_pth=model_path,
    batch_size =8
)
inferer.inference()