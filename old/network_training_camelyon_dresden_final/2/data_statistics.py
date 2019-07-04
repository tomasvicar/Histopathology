from histo_loader import HistoDataset
from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import torch.nn as nn

from torch.autograd import Variable
import unet


import visdom     #python -m visdom.server
viz=visdom.Visdom()

import drawloss
import time
import os
import torch.nn.functional as F
from PIL import Image



batch = 32
cn='no'
dataset='cam'


s=0
l=0
U=1
sm=0
lm=0
Um=1








Image.MAX_IMAGE_PIXELS = None


loader = HistoDataset(split='train',s=s,l=l,U=U,sm=sm,lm=lm,Um=Um,cn=cn,dataset=dataset)
trainloader= data.DataLoader(loader, batch_size=batch, num_workers=0, shuffle=True,drop_last=True)




kolik=1000


imgsum=0
lblsum=0
dataiter = iter(trainloader)
for k in range(kolik):
    print(k)
    (img,mask,lbl) = next(dataiter)
    
    lbl=mask[0]
    img=img[0]
    
    obr=img.data.numpy()
    lbl=lbl.data.numpy()
    imgsum+=np.sum(obr)
    lblsum+=np.sum(lbl)


img_mean=imgsum/(kolik*batch*np.shape(obr)[1]*np.shape(obr)[2]*np.shape(obr)[3])
lbl_mean=lblsum/(kolik*batch*np.shape(obr)[2]*np.shape(obr)[3])   

imgsum=0
lblsum=0
dataiter = iter(trainloader)
for k in range(kolik):
    print(k)
    (img,mask,lbl) = next(dataiter)
    
    lbl=mask[0]
    img=img[0]
    
    obr=img.data.numpy()
    lbl=lbl.data.numpy()
    
    imgsum+=np.sum(np.square(obr-img_mean))
    
std=np.sqrt(imgsum/(kolik*batch*np.shape(obr)[1]*np.shape(obr)[2]*np.shape(obr)[3]))    
    
print(img_mean)#161.6036
print(lbl_mean)#63.9972=>4.047=>3
print(std)#68.1900


