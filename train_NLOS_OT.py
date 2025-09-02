import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights
from torchvision.transforms import Normalize
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss

from dataset import ThermalNLOSDataset, RealDataset
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.NLOS_OT import Encoder, Decoder

import warnings
warnings.filterwarnings(
    "ignore",
    category=ResourceWarning,
    message="Implicitly cleaning up.*"
)

def save_model(model, path, name):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f"{name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved {name} at {save_path}")

def load_model(model, path, name, map_location=None):
    load_path = os.path.join(path, f"{name}.pth")
    state_dict = torch.load(load_path, map_location=map_location)
    model.load_state_dict(state_dict)
    print(f"Loaded {name} from {load_path}")
    return model

# Apply VGG pretrained features
class VGGFeature(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.norm = Normalize(mean=[0.5]*3, std=[0.5]*3)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # adapt to VGG features of 3 channels
        return self.vgg(self.norm(x))

## stage1:  train autoencoder (Encoder1, Decoder1)
def train_stage1(train_loader, encoder, decoder, epochs=10, lr=1e-4, λ=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder, decoder = encoder.to(device), decoder.to(device)
    vgg = VGGFeature().to(device)
    vgg.eval()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    loss_l1 = L1Loss()
    loss_mse = MSELoss()

    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        with tqdm(initial = 0, total = len(train_loader)) as pbar:
            for sample in train_loader:
                img=sample["img"].to(device) # shape: [B, 1, 128, 128]

                z = encoder(img)
                recon = decoder(z)

                # Perceptual Loss
                feat_orig = vgg(img)
                feat_recon = vgg(recon)
                perceptual_loss = loss_mse(feat_recon, feat_orig)

                # L1 Loss
                pixel_loss = loss_l1(recon, img)

                # Total Loss
                loss = perceptual_loss + λ * pixel_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(f'loss: {loss.item():.6f}')
                pbar.update(1)
        
        save_model(encoder, "Pth/nlos_ot", f"{epoch}encoder_E1")
        save_model(decoder, "Pth/nlos_ot", f"{epoch}decoder_D1")

        print(f"[Stage1] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


## stage2:  train Encoder2
def train_stage2(train_loader, E1, decoder, E2, epochs=10, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    E1 = E1.to(device)
    decoder = decoder.to(device)
    E2 = E2.to(device)

    E1.eval()     # frozen
    decoder.eval()

    optimizer = torch.optim.Adam(E2.parameters(), lr=lr)
    loss_fn = L1Loss()

    for epoch in range(epochs):
        E2.train()
        with tqdm(initial = 0, total = len(train_loader)) as pbar:
            for sample in train_loader:
                proj_img=sample["img"].to(device)
                gt_img=sample["gt"].to(device)

                with torch.no_grad():
                    z_gt = E1(gt_img)  # ground truth latent

                z_pred = E2(proj_img)
                loss = loss_fn(z_pred, z_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(f'loss: {loss.item():.6f}')
                pbar.update(1)

        save_model(E2, "Pth/nlos_ot", f"{epoch}encoder_E2")

        print(f"[Stage2] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def saveimg(data_list,name,results_folder="temp_results/"):
    data = torch.cat(data_list, dim=0)
    data=data.detach().cpu().numpy()
    for i in range(len(data)):
        img=np.squeeze(data[i,:,:])
        img=(img-np.min(img))/(np.max(img)-np.min(img))

        plt.imshow(img,cmap="gray")
        plt.savefig(results_folder+name+'{}.png'.format(i))

@torch.no_grad()
def test(val_loader, E2, decoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    E2 = E2.to(device).eval()
    decoder = decoder.to(device).eval()

    results=[]
    for batch_idx, data in enumerate(val_loader):
        z=E2(data["img"].to(device))
        pred_img = decoder(z)
        results.append(pred_img[:,0,:,:])
    
    saveimg(results,"img")

if __name__=="__main__":
    img_size=128
    crop_l=79
    batch_size=64

    # training_data_folder=["/data/NLOSFormer/TrainingData/T_dataset/","/data/NLOSFormer/TrainingData/T_dataset2/"]
    # training_data_folder=["/data/yrl/NLOSFormer/StandDataset2/","/data/yrl/NLOSFormer/OnehandDataset1/","/data/yrl/NLOSFormer/Left90","/data/yrl/NLOSFormer/Left130","/data/yrl/NLOSFormer/S_dataset/","/data/yrl/NLOSFormer/S_dataset2/","/data/yrl/NLOSFormer/T_dataset/","/data/yrl/NLOSFormer/T_dataset2/"]

    training_dataset=ThermalNLOSDataset("TrainingData/ThermalNLOSData/",name="base",mode='train',imgsize=img_size,crop_l=crop_l)

    # training_dataset=BlendDataset(training_data_folder,'train',ratio=0.8,imgsize=img_size,crop_l=crop_l)
    train_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

    ###
    ### Choose stage to perform 'stage1 train', 'stage2 train' or 'evaluate'
    ### 

    stage=3

    ## stage1 train
    if stage==1:
        epochs=5
        train_stage1(train_loader, Encoder(), Decoder(),epochs)

    ## stage2 train
    elif stage==2:
        best_epoch1=4 # This best .pt can be changed according to your cases
        encoder1 = load_model(Encoder(), "Pth/nlos_ot", f"{best_epoch1}encoder_E1")
        decoder1 = load_model(Decoder(), "Pth/nlos_ot", f"{best_epoch1}decoder_D1")
        train_stage2(train_loader, encoder1, decoder1,Encoder(),epochs=50)

    # evaluate
    elif stage==3:
        best_epoch1=4 # This best .pt can be changed according to your cases
        best_epoch2=49 # This best .pt can be changed according to your cases

        decoder1 = load_model(Decoder(), "Pth/nlos_ot", f"{best_epoch1}decoder_D1")
        encoder2 = load_model(Encoder(), "Pth/nlos_ot", f"{best_epoch2}encoder_E2")

        data=RealDataset("TestData/data_fig3_1/",img_size,False)
        val_loader= DataLoader(data, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())

        test(val_loader, encoder2, decoder1)