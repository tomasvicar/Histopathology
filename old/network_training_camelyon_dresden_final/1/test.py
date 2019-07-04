import numpy as np
import patcher_test
import torch
from torch.autograd import Variable
import os
from skimage.color import rgb2lab
from skimage.color import lab2rgb
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

from skimage.color import rgb2lab
from skimage.color import lab2rgb
import time

for data_type in ['test']:

    model_spec='1s_no_dense_net_s2_k012_k12_0.99015101510151__042600.pkl'
    model_name='1s_no_dense_net_s2_k012_k12'
    
    data_path='/media/ubmi/DATA1/vicar/cam/'+ data_type +'_cam/data' 
    cn='no'
    scale='1'
    dense_down=-1
    
    write_scale=1#wrtie smaler
    down_scale=16#network scale
    fg_scale_down=8#fg->img scale
    
#    step=320
#    border=0
    
    step=256*4
    border=120
    
    lvl=1

    
    auc_sampling_per=0.01*down_scale**2
    
    if auc_sampling_per>1:
        auc_sampling_per=1
    
    
    
    model_path='/home/ubmi/vicar/results/' +model_name+ '/models/' +model_spec
    save_folder='/home/ubmi/vicar/results/' +model_name  +'/' +data_type
    
    try:
        os.makedirs(save_folder)
    except:
        print('folder existuje')
    
    
    
         
        
    
    file_names=[]
    for root, dirs, files in os.walk(data_path):
        for name in files:
            if name.endswith(".tif"):
                file_names.append(root+'/'+name)
        
#        
    file_names=file_names[4:5]
    #file_names = [file_names[i] for i in [1,6]]
    
    model= torch.load(model_path)
    model=model.cuda()
    if dense_down>=0:
        model.dense_on(dense_down)
    model.eval()
    
    for_auc_res=[]
    for_auc_gt=[]
    TP=np.array(0,dtype=np.float64)
    TN=np.array(0,dtype=np.float64)
    FP=np.array(0,dtype=np.float64)
    FN=np.array(0,dtype=np.float64)
    
    img_sizes=[]
    
    
    for img_filename in file_names:
    
    
        fg_name=img_filename
        fg_name=fg_name.replace('/data/','/fg/')
        
        lbl_name=fg_name
        lbl_name=lbl_name.replace('/fg/','/mask/')
        lbl_name=lbl_name.replace('.tif','_mask.tif')
        
        tmp=img_filename.split('/')
        tmp=tmp[-1]
        save_name=save_folder +'/'+ tmp
        
        
        cn_path=data_path+'/'+ tmp[:-4] +'/'
        cn_path=cn_path.replace('_plice/data','patch_plice/test')
        

        
        patcher=patcher_test.Patcher(img_filename,fg_name,lbl_name,save_name,lvl=lvl,step=step,border=border,write_scale=write_scale,down_scale=down_scale,fg_scale_down=fg_scale_down)
        
        
        img_size=0
        for i in range(patcher.pocet_patchu()):
        

            img0,fg,lbl=patcher.read()
            
            
            imgs=[img0]
            
           
            if cn=='no':
                data_tmp=[]
                for img in imgs:
                    data_tmp.append(img/(255/2)-1)
                imgs=data_tmp
            
            
            imgs=imgs[0]
            
            img=np.float32(imgs)
            img=np.transpose(img,(2, 0, 1))
            img=torch.from_numpy(img).cuda()
            img=img.view(1,3,np.shape(img)[1],np.shape(img)[2])
            
            
            img=model(img)
            img=torch.sigmoid(img)
            img=img[0,0,:,:].data.cpu().numpy()
            
            
            
            q1=lbl[fg>0].flatten()
            q2=img[fg>0].flatten()
            if (len(q1)*auc_sampling_per)>0:
                r=np.random.choice(len(q1), size=int(len(q1)*auc_sampling_per), replace=False)
                for_auc_gt.append(q1[r])
                for_auc_res.append(q2[r])
            
            
            
            TP+=np.sum(np.bitwise_and(np.bitwise_and(lbl>0,img>0.5),fg>0))
            TN+=np.sum(np.bitwise_and(np.bitwise_and(lbl==0,img<=0.5),fg>0))
            FP+=np.sum(np.bitwise_and(np.bitwise_and(lbl==0,img>0.5),fg>0))
            FN+=np.sum(np.bitwise_and(np.bitwise_and(lbl>0,img<=0.5),fg>0))
            
            
            img_size=img_size+np.sum(fg>0)
            img[fg==0]=0

            patcher.write(img)
        patcher.close()
        img_sizes.append(img_size)
    
       
    
    for_auc_res=np.concatenate(for_auc_res,axis=0)        
    for_auc_gt=np.concatenate(for_auc_gt,axis=0)      
    
    
    fpr, tpr, thresholds = roc_curve(for_auc_gt>0,for_auc_res,pos_label=1)
    
    auc_s=auc(fpr, tpr)
    sen=TP/(TP+FN)
    spe=TN/(TN+FP)
    acc=(TP+TN)/(TP+FP+FN+TN)
    dice=2*TP/(2*TP+FP+FN)
    
    plt.plot(fpr,tpr)
    plt.show()
    plt.plot(fpr,tpr)
    plt.savefig(save_folder +'/' + 'roc.png')
    
    plt.clf()
    
    file = open(save_folder +'/' + 'results.txt','w') 
    file.write('TP ' + str(TP) +'\n')
    file.write('TN ' + str(TN)+'\n')
    file.write('FP ' + str(FP)+'\n')
    file.write('FN ' + str(FN)+'\n')
    file.write('auc ' + str(auc_s)+'\n')
    file.write('sen ' + str(sen)+'\n')
    file.write('spe ' + str(spe)+'\n')
    file.write('acc ' + str(acc)+'\n')
    file.write('dice ' + str(dice)+'\n')
    file.close()
    time.sleep(0.01)
    
    
    print('TP ' + str(TP))
    print('TN ' + str(TN))
    print('FP ' + str(FP))
    print('FN ' + str(FN))
    print('auc ' + str(auc_s))
    print('sen ' + str(sen))
    print('spe ' + str(spe))
    print('acc ' + str(acc))
    print('dice ' + str(dice))








