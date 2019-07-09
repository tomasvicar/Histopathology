import torch
import numpy as np
import time
import os




def l1_loss(input, target):
    if torch.Tensor==type(input):
        return torch.mean(torch.abs(input - target))
    else:
        return np.mean(np.abs(input - target))

def l2_loss(input, target):
    if torch.Tensor==type(input):
        return torch.mean((input - target)**2)
    else:
        return np.mean((input - target)**2)

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
    def __init__(self,lr_step=10_000,stopcount_limit=7,drop_factor=0.5):
        self.lr_step=lr_step##if no improvement
        self.best_loss=999999999
        self.best_loss_pos=0
        self.stopcount=0
        self.stopcount_limit=stopcount_limit
        self.drop_factor=drop_factor
        
    def step(self,optimizer,iteration=0,loss=0):

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
            self.best_loss_pos=iteration
            self.best_loss=loss
#        
        print(self.best_loss,loss)
#        
#        print(str(self.batch*iteration-self.best_loss_pos) + '////' + str(self.lr_step))
        if  iteration-self.best_loss_pos>self.lr_step:
            self.stopcount+=1
            self.best_loss_pos=iteration
            self.best_loss=loss
            print('lr down')
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] *self.drop_factor
                
        if  self.stopcount>=self.stopcount_limit:
            return 1
        else:
            return 0