import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import scipy
import numpy as np
import matplotlib.pyplot as plt

import os
from multiprocessing import cpu_count

class Inferer(object):
    def __init__(
        self,
        dataset,
        model,
        pca_encoder,
        model_pth,
        batch_size = 16,
        result_folder="results/"
    ):
        super().__init__()

        self.dl= DataLoader(dataset, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = cpu_count())

        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # Load parameters
        if model_pth!="":
            data = torch.load(model_pth,map_location=self.device)
            model.load_state_dict(data['model'])
        self.model=model.to(self.device)

        self.pca_encoder=pca_encoder
        self.pca_encoder.initialize(self.device)

        # output dir
        self.results_folder =result_folder
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

    def inference(self):
        results=[]

        with torch.no_grad():
            self.model.eval()
            for batch_idx, data in enumerate(self.dl):
                pred_img,reduced_kernel=self.model(data["img"].to(self.device)) ## (batch,channel,height,width)
                results.append(pred_img)

        self._saveimg(results,"img")
        
    def _saveimg(self,data,name):
        data = torch.cat(data, dim=0)
        data=data.detach().cpu().numpy()
        for i in range(len(data)):
            img=np.squeeze(data[i,:,:])
            img=(img-np.min(img))/(np.max(img)-np.min(img))
            plt.imshow(img,cmap="gray")
            plt.savefig(self.results_folder+name+'{}.png'.format(i))
