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
import dense_net_pixel256 as pixel_net


import visdom     #python -m visdom.server
viz=visdom.Visdom()

import drawloss
import time
import os
import torch.nn.functional as F


#16batch,20000it = 32000
##lr_step - ->auto
iterace=99999999999999999
lr_step=100000
init_lr = 0.001
hard_neg_step=iterace
batch = 16 #64/16
k0=12
k=12
cn='no'
model_name='1s_no_dense_net_s3_k012_k12'
net='P'
gconv=0
path_to_data='/home/ubmi/vicar/patches_0s'
save_dir='/home/ubmi/vicar/results'



input_layers=3


s1=0
s2=0
s3=1
s1m=0
s2m=0
s3m=0



try:
    os.makedirs(save_dir + '/'+ model_name+  '/models')
except:
    pass



def dice_loss_logit(pred, target):
  
    pred=torch.sigmoid(pred)
    smooth = 1.

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )





class AdjustLearningRate():
    def __init__(self,lr_step,batch):
        self.lr_step=lr_step
        self.batch=batch
        self.best_loss=999999999
        self.best_loss_pos=0
        self.stopcount=0
        
    def step(self,optimizer,iteration,loss):

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
        
        if  loss<self.best_loss:
            self.best_loss_pos=iteration*batch
            self.best_loss=loss
        
        print(self.best_loss,loss)
        
        print(str(self.batch*iteration-self.best_loss_pos) + '////' + str(self.lr_step))
        if self.batch*iteration-self.best_loss_pos>self.lr_step:
            self.stopcount+=1
            self.best_loss_pos=iteration*batch
            self.best_loss=loss
            print('lr down')
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] *0.5
                
        if  self.stopcount>=6:
            return 1
        else:
            return 0
                
                

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




loader = HistoDataset(split='train',path_to_data=path_to_data,s1=s1,s2=s2,s3=s3,s1m=s1m,s2m=s2m,s3m=s3m,cn=cn)
trainloader= data.DataLoader(loader, batch_size=batch, num_workers=0, shuffle=True,drop_last=True)

loader = HistoDataset(split='valid',path_to_data=path_to_data,s1=s1,s2=s2,s3=s3,s1m=s1m,s2m=s2m,s3m=s3m,cn=cn)
validloader= data.DataLoader(loader, batch_size=batch, num_workers=0, shuffle=False,drop_last=False)



hard_negative=0

stop=0


if net=='U':
    model=unet.Unet(feature_scale=8,input_size=input_layers)
if net=='P':
    model=pixel_net.PixelNet(k0=k0,k=k,input_size=input_layers,gconv=gconv)


model=model.cuda()

optimizer = optim.Adam(model.parameters(),lr = init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)

display_losses= drawloss.DisplayLosses(evaluate_it=25)



lrad=AdjustLearningRate(lr_step,batch)

valid_loss=999999999

itt=-1
while itt<iterace and stop==0:
    for it,(img,mask,lbl) in enumerate(trainloader):
        itt+=1

        img=img[0]
        if net == 'U': 
            lbl=mask[0]
        
        
        
        
        hard_negative=adjust_hard_negative(hard_negative,itt,hard_neg_step)
        

        model.train()
        
        
        img = Variable(img.cuda(0))
        lbl = Variable(lbl.cuda(0))
        
        output=model(img)
        
        
        optimizer.zero_grad()
        

        if net == 'U':      
            loss = dice_loss_logit(output,lbl.float())
            
           
#            w=8.327-1 ##lungs
#            loss = F.binary_cross_entropy_with_logits(output,lbl.float(),weight=((lbl.float()*w)+1))  
                
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
                    loss = dice_loss_logit(output,lbl.float())


#                    w=8.327-1##lungs - now you should measure
#                    loss = F.binary_cross_entropy_with_logits(output,lbl.float(),weight=((lbl.float()*w)+1))  
                        
                        
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
            
            stop=lrad.step(optimizer,itt,display_losses.get_loss_test_last())
            
            torch.save(model,save_dir + '/' +model_name+ '/models/' +model_name  +'_' + str(display_losses.get_auc_test_last())+ '__' + str(itt).zfill(6)+'.pkl')
            
            display_losses.save_plots(save_dir + '/' +model_name)
        
            tr,te=display_losses.get_data()
            
            np.save(save_dir + '/' +model_name +'/training_log_train.npy', tr)
            np.save(save_dir + '/' +model_name +'/training_log_test.npy', te)
            
       
            
            
