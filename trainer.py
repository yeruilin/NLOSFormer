"""Define Trainer: define the updating process"""
from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from ema_pytorch import EMA
from pathlib import Path
from multiprocessing import cpu_count
from einops import rearrange

import scipy
import numpy as np
import matplotlib.pyplot as plt

import os
from tqdm import tqdm

from utils import *
from loss import *


class Trainer(object):
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        PCAEncoder,
        *,
        train_batch_size = 16,
        epoch=16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        split_batches = True,
        save_epoch=200,
        result_folder="pth/"
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'no'
        )
        
        # model
        self.model = model

        self.batch_size = train_batch_size
        self.epoch=epoch
        self.save_epoch=save_epoch
        self.gradient_accumulate_every = gradient_accumulate_every

        # dataset and dataloader
        train_dl = DataLoader(train_dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        self.train_dl = self.accelerator.prepare(train_dl)

        val_dl = DataLoader(val_dataset, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())
        self.val_dl = self.accelerator.prepare(val_dl)

        # optimizer
        self.opt = Adam(model.parameters(), lr = train_lr, betas = adam_betas)
        
        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.pca_encoder=PCAEncoder
        self.pca_encoder.initialize(self.device)

        self.results_folder =result_folder
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, self.results_folder+str(f'model-{milestone}.pt'))

    def load(self, ckpt):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(ckpt, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])


        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # Train function
    def train(self, mode="NLOSFormer"):
        accelerator = self.accelerator
        device = accelerator.device

        total_loss_list=[]
        if mode=="NLOSFormer":
            criterion=MyLoss()
        elif mode=="LMSNet":
            criterion=MSLoss()
        else:
            criterion=RestoreLoss()

        for i in range(self.epoch):
            with tqdm(initial = 0, total = len(self.train_dl), disable = not accelerator.is_main_process) as pbar:
                total_loss = 0.0
                for sample in self.train_dl:
                    img=sample["img"].to(device)
                    psf=sample["psf"].to(device)
                    gt=sample["gt"].to(device)

                    with self.accelerator.autocast():
                        if mode=="NLOSFormer":
                            pred_img,pred_kernel = self.model(img) # pred_img,gt is (batch,channel,height,width)
                            reduced_psf=self.pca_encoder.encode(psf)
                            loss=criterion(pred_img,gt,pred_kernel,reduced_psf)
                        elif mode=="LMSNet":
                            pred_list = self.model(img)
                            loss=criterion(pred_list,gt)
                        else:
                            pred_img = self.model(img)
                            loss=criterion(pred_img,gt)
                        
                        total_loss = loss.item()

                    self.accelerator.backward(loss)

                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    pbar.set_description(f'loss: {total_loss:.6f}')

                    total_loss_list.append(total_loss)

                    accelerator.wait_for_everyone()

                    # for name, param in self.model.named_parameters():
                    #     if param.grad is None:
                    #         print(name)

                    self.opt.step()
                    self.opt.zero_grad()

                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        self.ema.update() # this variable `self.ema` is only defined in main process

                    pbar.update(1)

            # validation
            with torch.no_grad():
                self.model.eval()
                val_loss = 0.0
                cnt=0

                for sample in tqdm(self.val_dl):
                    img=sample["img"].to(device)
                    psf=sample["psf"].to(device)                   
                    gt=sample["gt"].to(device)
                    with self.accelerator.autocast():
                        if mode=="NLOSFormer":
                            pred_img,pred_kernel = self.model(img)
                            reduced_psf=self.pca_encoder.encode(psf)
                            loss=criterion(pred_img,gt,pred_kernel,reduced_psf)
                        elif mode=="LMSNet":
                            pred_list = self.model(img)
                            loss=criterion(pred_list,gt)
                        else:
                            pred_img = self.model(img)
                            loss=criterion(pred_img,gt)
                        
                        val_loss+= loss.item()
                        
                        cnt+=1

            if accelerator.is_main_process:
                print("validation loss: {0},{1}".format(val_loss/cnt,val_loss/cnt))
                self.ema.ema_model.eval()
                with torch.no_grad():
                    self.save(i)
            
        accelerator.print('training complete')
        scipy.io.savemat('train_loss.mat', {'loss': np.array(total_loss_list)})
        
        accelerator.end_training()

class Evaluator(object):
    def __init__(
        self,
        model,
        val_dataset,
        train_batch_size = 16
    ):
        super().__init__()

        self.accelerator = Accelerator(split_batches = True, mixed_precision = 'no')
        
        # model
        self.model = model

        self.batch_size = train_batch_size
    
        # dataset and dataloader
        val_dl = DataLoader(val_dataset, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())
        self.val_dl = self.accelerator.prepare(val_dl)

        self.model = self.accelerator.prepare(self.model)

    @property
    def device(self):
        return self.accelerator.device
    
    def load(self, ckpt):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(ckpt, map_location=device)
        
        if "model" in data.keys():
            data=data["model"]

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data)

    def evaluate(self, mode="NLOSFormer"):
        accelerator = self.accelerator
        device = accelerator.device
        
        criterion=nn.MSELoss()

        with torch.no_grad():
            self.model.eval()
            val_loss = 0.0
            cnt=0

            for sample in tqdm(self.val_dl):
                img=sample["img"].to(device)             
                gt=sample["gt"].to(device)
                with self.accelerator.autocast():
                    if mode=="NLOSFormer":
                        pred_img,pred_kernel = self.model(img)
                        loss=criterion(pred_img,gt)
                    elif mode=="LMSNet":
                        pred_list = self.model(img)
                        loss=criterion(pred_list[-1],gt)
                    else:
                        pred_img = self.model(img)
                        loss=criterion(pred_img,gt)
                    
                    val_loss+= loss.detach()
                    
                    cnt+=img.size(0)

            val_loss = accelerator.reduce(val_loss, reduction="sum")

            if accelerator.is_main_process:
                avg_val_loss = (val_loss / cnt/4).item()
                print(f"[Validation] Average Loss: {avg_val_loss:.6f}")
        
        accelerator.end_training()

class Inferer(object):
    def __init__(
        self,
        dataset,
        model,
        pca_encoder,
        model_pth,
        batch_size = 16,
        result_folder="temp_results/"
    ):
        super().__init__()

        self.dl= DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size=batch_size

        # # Read model
        if model_pth!="":
            data = torch.load(model_pth,map_location=self.device)
            if "model" in data.keys():
                data=data['model']
            model.load_state_dict(data)
        self.model=model.to(self.device)

        self.pca_encoder=pca_encoder
        self.pca_encoder.initialize(self.device)

        # Output folder
        self.results_folder =result_folder
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

    def inference(self,mode="NLOSFormer"):
        results=[]
        results_kernel=[]

        with torch.no_grad():
            self.model.eval()
            for batch_idx, data in enumerate(self.dl):
                if mode=="NLOSFormer":
                    pred_img,reduced_kernel=self.model(data["img"].to(self.device)) ## (batch,channel,height,width)
                    kernel=self.pca_encoder.decode(reduced_kernel) # batch, crop_l, crop_l
                    results.append(pred_img[:,0,:,:])
                    results_kernel.append(kernel)
                elif mode=="LMSNet":
                    pred_list = self.model(data["img"].to(self.device))
                    pred_img=pred_list[-1]
                    results.append(pred_img[:,0,:,:])
                else:
                    pred_img=self.model(data["img"].to(self.device))
                    results.append(pred_img[:,0,:,:])

        if mode=="NLOSFormer":
            self._saveimg(results,"img")
            # self._saveimg(results_kernel,"kernel")
        else:
            self._saveimg(results,"img")
        
    def _saveimg(self,data,name):
        data = torch.cat(data, dim=0)
        data=data.detach().cpu().numpy()
        for i in range(len(data)):
            img=np.squeeze(data[i,:,:])
            img=(img-np.min(img))/(np.max(img)-np.min(img))
            # scipy.io.savemat(self.results_folder+name+'{}.mat'.format(i),{"img":img}) # save results
            plt.imshow(img,cmap="jet")
            plt.savefig(self.results_folder+name+'{}.png'.format(i))

            # scipy.io.savemat(self.results_folder+'kernel{}.mat'.format(i), {'kernel': data})
    
    def predict(self,img_tensor,mode="NLOSFormer"):
        # img_tensor: [H,W,frames]
        results=[]
        results_kernel=[]
        
        frame_num=img_tensor.shape[2]
        epochs=frame_num//self.batch_size
        
        img_tensor=img_tensor.permute(2,0,1)

        with torch.no_grad():
            self.model.eval()
            for i in range(epochs):
                input=img_tensor[i*self.batch_size:(i+1)*self.batch_size,:,:]
                input=input[:,None,:,:].to(self.device) ## (batch,channel,height,width)
                if mode=="NLOSFormer":
                    pred_img,reduced_kernel=self.model(input) ## (batch,channel,height,width)
                    kernel=self.pca_encoder.decode(reduced_kernel) # batch, crop_l, crop_l
                    results.append(pred_img[:,0,:,:])
                    results_kernel.append(kernel)
                elif mode=="LMSNet":
                    pred_list = self.model(input)
                    pred_img=pred_list[-1]
                    results.append(pred_img[:,0,:,:])
                else:
                    pred_img=self.model(input)
                    results.append(pred_img[:,0,:,:])

        if mode=="NLOSFormer":
            self._saveimg(results,"img")
            # self._saveimg(results_kernel,"kernel")
        else:
            self._saveimg(results,"img")