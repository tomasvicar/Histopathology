from histo_loader import HistoDataset
from torch.utils import data
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pixel_net_eq256_kplus as pixel_net
import torch.nn.functional as F
from torch import optim
from utils.training_function import l1_loss,l2_loss,dice_loss_logit,AdjustLearningRate
#import torch.multiprocessing as mp
#mp.set_start_method('spawn') 



iters=9999999999
batch=16
init_lr=0.01
path_to_data_train='/media/ubmi/DATA2/vicar/cam_dataset/train/data'
path_to_data_valid='/media/ubmi/DATA2/vicar/cam_dataset/valid/data'
k0=16
k=8
gconv=0
lvl=1
valid_freq=300 ##iters
lr_step=500_000/batch ##iters

save_dir='../results/s1_pixel_baseinfsampler'


try:
    os.makedirs(save_dir)
except:
    pass



#def worker_init_fn(worker_id):                                                          
#    np.random.seed(np.random.get_state()[1][0] + worker_id)
    




if __name__ == '__main__':

    
        
    loader = HistoDataset(split='train',path_to_data=path_to_data_train,level=lvl)
    trainloader= data.DataLoader(loader, batch_size=batch, num_workers=3, shuffle=True,drop_last=True,pin_memory=False,worker_init_fn=None)
    
    loader = HistoDataset(split='valid',path_to_data=path_to_data_valid,level=lvl)
    validloader= data.DataLoader(loader, batch_size=batch, num_workers=3, shuffle=False,drop_last=False,pin_memory=False)
    
    def inf_train_gen():
        while True:
#            np.random.seed()
            for img,mask,lbl in trainloader:
                yield img,mask,lbl
                
    gen = inf_train_gen()
    
#    model=pixel_net.PixelNet(K=k0,kplus=k,input_size=3,gconv=gconv)
    model=torch.load('/media/ubmi/DATA1/vicar/code/results/s1_pixel_baseinfsampler/0.9665__00055500.pkl')
    model=model.cuda(0)
    
    optimizer = optim.Adam(model.parameters(),lr = init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    adjustLerningRate=AdjustLearningRate(lr_step=lr_step,stopcount_limit=7,drop_factor=0.5)
    
    it=0
    stop=0
    loss_train=[]
    acc_train=[]
    it_train=[]
    loss_valid=[]
    acc_valid=[]
    it_valid=[]
    losses=[]
    accs=[]
    while it<iters and stop==0:
        it=it+1
#        print(it)
        
        img,mask,lbl=next(gen)
        
        img=img.cuda(0)
        mask=mask.cuda(0)
        lbl=lbl.cuda(0)
        
#        print(lbl[0])
#        plt.imshow(np.concatenate((img[0,0,:,:].data.cpu().numpy(),mask[0,0,:,:].data.cpu().numpy()),axis=1))
#        plt.show()
#        
        img.requires_grad=True
        mask.requires_grad=True
                
        model.train()

        output=model(img)
        
        loss = F.binary_cross_entropy_with_logits(output.squeeze(),lbl.squeeze())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        clasif=torch.sigmoid(output)
        
        lbl=lbl.view(-1).detach().cpu().numpy()
        clasif=clasif.view(-1).detach().cpu().numpy()
        acc=np.sum(lbl==(clasif>0.5))/clasif.shape[0]
                
        loss=loss.detach().cpu().numpy()
        losses.append(loss)
        accs.append(acc)
            
        
        if it % valid_freq == 0:
            
            loss_train.append(np.mean(losses))
            acc_train.append(np.mean(accs))
            it_train.append(it*batch)
            
            losses=[]
            accs=[]
            for itt,(img,mask,lbl) in enumerate(validloader):
                img=img.cuda(0)
                mask=mask.cuda(0)
                lbl=lbl.cuda(0)
                
                model.eval()

                output=model(img)
                
                loss = F.binary_cross_entropy_with_logits(output.squeeze(),lbl.squeeze())
                
                clasif=torch.sigmoid(output)
                
                lbl=lbl.view(-1).detach().cpu().numpy()
                clasif=clasif.view(-1).detach().cpu().numpy()
                acc=np.sum(lbl==(clasif>0.5))/clasif.shape[0]

                loss=loss.detach().cpu().numpy()
                losses.append(loss)
                accs.append(acc)
                
            loss_valid.append(np.mean(losses))
            acc_valid.append(np.mean(accs))
            it_valid.append(it*batch)
            
            stop=adjustLerningRate.step(optimizer,it,loss_valid[-1])
            
            losses=[]
            accs=[]
            
            torch.save(model,save_dir + os.sep + str(acc_valid[-1])+ '__' + str(it).zfill(8)+'.pkl')
            
            np.savez(save_dir +os.sep+'training_log_train.npy', np.array(it_train),np.array(it_valid),np.array(loss_train),np.array(loss_valid))
            np.savez(save_dir +os.sep+'training_log_test.npy', np.array(it_train),np.array(it_valid),np.array(acc_train),np.array(acc_valid))
            
            
            for param_group in optimizer.param_groups:
                lr_act=param_group['lr']
            
            print(str(it) + ' train loss: ' + str(loss_train[-1]) +  '  train ACC: ' +str(acc_train[-1])
            + ' test loss: ' + str(loss_valid[-1]) +  '  test ACC: ' +str(acc_valid[-1]) + '  lr: ' +str(lr_act))
            
            plt.plot(it_train,loss_train)
            plt.plot(it_valid,loss_valid)
            plt.show()
            
            plt.plot(it_train,acc_train)
            plt.plot(it_valid,acc_valid)
            plt.show()
        
        
            plt.plot(it_train,loss_train)
            plt.plot(it_valid,loss_valid)
            plt.savefig(save_dir +os.sep+ 'loss.png')
            
            plt.plot(it_train,acc_train)
            plt.plot(it_valid,acc_valid)
            plt.savefig(save_dir +os.sep+ 'acc.png')
        
        
        
        
        
        
        
        
        
        
       