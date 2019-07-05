from histo_loader import HistoDataset
from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


path_to_data='/media/ubmi/DATA2/vicar/cam_dataset/train/data'

batch = 1
loader = HistoDataset(split='train',path_to_data=path_to_data,level=1)
trainloader= data.DataLoader(loader, batch_size=batch, num_workers=1, shuffle=True,drop_last=True,pin_memory=False)


for it,(img,mask,lbl) in enumerate(trainloader):
    
    tmp=img.data.cpu().numpy()
    tmp=tmp[0,0,:,:]
    plt.imshow(tmp)
    plt.show()
    
    tmp=mask.data.cpu().numpy()
    tmp=tmp[0,0,:,:]
    plt.imshow(tmp)
    plt.show()
    print(lbl)
    
    a=5