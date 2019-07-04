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
import pixel_net as pixel_net


import visdom     #python -m visdom.server
viz=visdom.Visdom()

import drawloss
import time
import os
import torch.nn.functional as F
from PIL import Image




iterace=60000
lr_step=15000
init_lr = 0.01
hard_neg_step=iterace
batch = 32
net_K=80
dataset='cam'
cn='no'
model_name='1s_no_pixel_cir'
net='P'


input_layers=3


s=1
l=0
U=0
sm=0
lm=0
Um=0











try:
    os.mkdir('..\\results\\' +model_name)
except:
    pass
try:
    os.mkdir('..\\results\\' +model_name+ '\\models')
except:
    pass










Image.MAX_IMAGE_PIXELS = None




def dice_loss_logit(pred, target):
  
    pred=F.sigmoid(pred)
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )







def clear_border(maskin,n):
    maskout=maskin
    maskout[0:n,:]=False
    maskout[-n:,:]=False
    maskout[:,0:n]=False
    maskout[:,-n:]=False
    return maskout



def adjust_learning_rate(optimizer,iteration,lr_step):

    try:
        with open('lr_change.txt', 'r') as f:
            x = f.readlines()
            lr=float(x[0])
        time.sleep(1)
        os.remove('lr_change.txt')
        
        print('lr was set to: ' + str(x))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    except:
        pass
        
    if iteration%lr_step==0 and iteration!=0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] *0.1

def adjust_hard_negative(hard_negative,iteration,hard_neg_step):
    try:
        with open('hn_change.txt', 'r') as f:
            x = f.readlines()
            lr=float(x[0])
        time.sleep(1)
        os.remove('hn_change.txt')
        
        print('lr was set to: ' + str(x))
        hard_negative=lr
    except:
        pass
        
    if iteration%hard_neg_step==0 and iteration!=0:
        hard_negative=hard_negative+0.2
        
    return hard_negative




loader = HistoDataset(split='train',s=s,l=l,U=U,sm=sm,lm=lm,Um=Um,cn=cn,dataset=dataset)
trainloader= data.DataLoader(loader, batch_size=batch, num_workers=0, shuffle=True,drop_last=True)

loader = HistoDataset(split='valid',s=s,l=l,U=U,sm=sm,lm=lm,Um=Um,cn=cn,dataset=dataset)
validloader= data.DataLoader(loader, batch_size=batch, num_workers=0, shuffle=False,drop_last=False)



hard_negative=0


if net=='U':
    model=unet.Unet(feature_scale=8,input_size=input_layers)
if net=='P':
    model=pixel_net.PixelNet(K=net_K,input_size=input_layers)


model=model.cuda()

optimizer = optim.Adam(model.parameters(),lr = init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-5)


display_losses= drawloss.DisplayLosses(evaluate_it=5)

itt=-1
while itt<iterace:
    for it,(img,mask,lbl) in enumerate(trainloader):
        itt+=1

        img=img[0]
        if net == 'U': 
            lbl=mask[0]
        
        
        adjust_learning_rate(optimizer,itt,lr_step)
        
        hard_negative=adjust_hard_negative(hard_negative,itt,hard_neg_step)
        

        model.train()
        
        
        img = Variable(img.cuda(0))
        lbl = Variable(lbl.cuda(0))
        

        output=model(img)
        
                

        optimizer.zero_grad()
        

        if net == 'U':      
#            loss = dice_loss_logit(output,lbl.float())
            if dataset=='cam':
                w=2.688-1
            
            loss = F.binary_cross_entropy_with_logits(output,lbl.float(),weight=((lbl.float()*w)+1))    
        if net == 'P':
            loss = F.binary_cross_entropy_with_logits(output.squeeze(),lbl.float())
        
            
        
        loss.backward()
        optimizer.step()
        
        clasif=torch.sigmoid(output)
        
        if net=='U':
            lbl=lbl[...,159,159]
            clasif=clasif[...,159,159]
        
        loss=loss.data.cpu().numpy()
        lbl=lbl.data.cpu().numpy()
        clasif=clasif.data.cpu().numpy()
        
        
        display_losses.update_train(loss,lbl,clasif,optimizer.param_groups[0]['lr'])
        

        
        if itt%200==0:
            val_it=0
            for itv,(img,mask,lbl) in enumerate(validloader):
                print(itv)
                val_it+=1
                
                
                img=img[0]
                if net == 'U': 
                    lbl=mask[0]
                

                model.eval()
                
                
                
                img = Variable(img.cuda(0))

                lbl = Variable(lbl.cuda(0))
                
         
                
                output=model(img)
                
                if net == 'U':      
#                    loss = dice_loss_logit(output,lbl.float())
                    loss = F.binary_cross_entropy_with_logits(output,lbl.float(),weight=((lbl.float()*4.94)+1))    
                if net == 'P':
                    loss = F.binary_cross_entropy_with_logits(output.squeeze(),lbl.float())
        
                clasif=torch.sigmoid(output)
                
                if net=='U':
                    lbl=lbl[...,159,159]
                    clasif=clasif[...,159,159]

                loss=loss.data.cpu().numpy()
                lbl=lbl.data.cpu().numpy()
                clasif=clasif.data.cpu().numpy()
                
                display_losses.update_test(loss,lbl,clasif)


                

            display_losses.draw_test()
            
            torch.save(model,'..\\results\\' +model_name+ '\\models\\' +model_name + str(itt).zfill(6)  +'_' + str(display_losses.get_auc_test_last())+'.pkl')
            
            display_losses.save_plots('..\\results\\' +model_name)
        
            tr,te=display_losses.get_data()
            
            np.save('..\\results\\' +model_name +'\\training_log_train.npy', tr)
            np.save('..\\results\\' +model_name +'\\training_log_test.npy', te)
            
            
       
            
            
