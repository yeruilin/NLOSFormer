# -*- coding: utf-8 -*-

from model.NLOSFormer import NLOSFormer
from PCAEncoder import PCAEncoder
from dataset import *
from trainer import Inferer

# parameters
img_size=128
crop_l=79
pca_path="preprocess/pca_matrix42.mat"

# load model
model=NLOSFormer(input_channel=1,hidden_channel=64,psf_dim=42)
model_path="Pth/pth_nlosformer/model-03302340.pt"

pca_encoder=PCAEncoder(pca_path,l=img_size, crop_l=79, dim=42,average=True)

# inference
simudataset=RealDataset("TestData/squat/",img_size,False)

inferer = Inferer(
    simudataset,
    model,
    pca_encoder,
    model_pth=model_path,
    batch_size =8
)
inferer.inference()