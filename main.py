# -*- coding: utf-8 -*-
# accelerate launch main.py --num_processes=4

import torch
from scipy.io import loadmat

from PCAEncoder import PCAEncoder
from dataset import *
from trainer import Trainer, Evaluator, Inferer
import argparse

from model.unet import UNet
from model.LMS_NLOS import LMSNet
from model.MS_NLOS import MS_NLOS
from model.NLOS_I2V import NLOS_I2V
from model.NLOS_OT import NLOS_OT

import warnings
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message="Implicitly cleaning up.*"
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NLOSFormer_v2", type=str, help="NLOSFormer/NLOSFormer_v2/UNet/NLOS_OT/NLOS_I2V/LMSNet")
    parser.add_argument("--stage", default="test", type=str,help="train/evaluate/test")
    parser.add_argument("--data_path", default="TestData/sample_data/", type=str, help="the path of training or testing data")
    parser.add_argument("--pth", default="model-augment.pt", type=str, help="the path of pretrained weights")
    parser.add_argument("--augment", default=True, type=bool, help="whether to use data augmentation")
    parser.add_argument("--kernel", default=True, type=bool, help="whether to use kernel supervision")
    args = parser.parse_args()
    return args

"""# Training Hyper-parameters"""

torch.backends.cudnn.benchmark = True
torch.manual_seed(4096)

if torch.cuda.is_available():
    torch.cuda.manual_seed(4096)

img_size=128
crop_l=79
batch_size = 8
epoch = 10
lr = 1e-3
grad_steps = 1
ema_decay = 0.995 # exponential moving average decay
pca_dim=42 # the dimension of the reduced kernel

args=get_args()

mode=args.model

if args.kernel:
    mode="NLOSFormerNK" # NLOSFormer with kernel supervision
if args.augment:
    data_name="augment"
else:
    data_name="base"

if mode=="NLOSFormer":
    from model.NLOSFormer import NLOSFormer
    model=NLOSFormer(input_channel=1,hidden_channel=64,psf_dim=pca_dim)
    pth_folder="Pth/pth_nlosformer/"

elif mode=="NLOSFormer_v2":
    from model.NLOSFormer_v2 import NLOSFormer
    model=NLOSFormer(input_channel=1,hidden_channel=64,psf_dim=pca_dim)
    pth_folder="Pth/pth_nlosformer_v2/"
    mode="NLOSFormer"

elif mode=="NLOSFormerNK":
    from model.NLOSFormerNK import NLOSFormerNK
    model=NLOSFormerNK(input_channel=1,hidden_channel=64)
    pth_folder="Pth/pth_nlosformerNK/"
  
elif mode=="UNet":
    model=UNet()
    pth_folder="Pth/pth_unet/"

elif mode=="LMSNet":
    model=LMSNet()
    # model=MS_NLOS()
    pth_folder="Pth/pth_lmsnet/"

elif mode=="NLOS_I2V":
    model=NLOS_I2V()
    pth_folder="Pth/pth_nlosi2v/"

elif mode=="NLOS_OT":
    model=NLOS_OT()
    pth_folder="Pth/nlos_ot/"

# PCA matrix
pca_path=f"./preprocess/pca_matrix{pca_dim}.mat"
pca_encoder=PCAEncoder(pca_path,l=img_size, crop_l=crop_l, dim=pca_dim)

stage=args.stage

if stage=="train":
    training_dataset=ThermalNLOSDataset(args.data_path,name=data_name,mode='train',imgsize=img_size,crop_l=crop_l)
    val_dataset=ThermalNLOSDataset(args.data_path,name=data_name,mode='val',imgsize=img_size,crop_l=crop_l)

    trainer = Trainer(
        model,
        training_dataset,
        val_dataset,
        pca_encoder,
        train_batch_size = batch_size,
        epoch=epoch,
        train_lr = lr,
        gradient_accumulate_every = grad_steps,
        ema_decay = ema_decay,
        pth_folder=pth_folder
    )
    trainer.train(mode=mode)

elif stage=="test":
    testdataset=MatDataset(args.data_path,img_size,False)
    inferer = Inferer(
        testdataset,
        model,
        pca_encoder,
        model_pth=pth_folder+args.pth,
        batch_size =8
    )
    inferer.inference(mode)

elif stage=="evaluate":
    #### Evaluate
    eval_dataset=ThermalNLOSDataset(args.data_path,name='test',imgsize=img_size,crop_l=crop_l)
    evaluator=Evaluator(model,eval_dataset,batch_size)
    evaluator.load(pth_folder+args.pth)
    evaluator.evaluate(mode)