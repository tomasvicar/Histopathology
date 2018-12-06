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
import cv2
import gdal
import augmenter

from skimage.io import imsave



def img_with_cont(img,mask,c=(0,0,255),th=1):
    
    mask=np.uint8((np.squeeze(mask)>0)*255)
    mask=np.squeeze(mask)
    
    
    img=np.array(img,dtype=np.uint8)
    
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, c, th)
    return img




def imread_gdal(data_name,level):
    
    level=level-1
    
    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    gOverview = gdalObj.GetRasterBand (2)
    bOverview = gdalObj.GetRasterBand (3)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        gOverview = gOverview.GetOverview(level)
        bOverview = bOverview.GetOverview(level)
    rOverview = rOverview.ReadAsArray(0,0, rOverview.XSize, rOverview.YSize) 
    gOverview = gOverview.ReadAsArray(0,0, gOverview.XSize, gOverview.YSize)  
    bOverview = bOverview.ReadAsArray(0,0, bOverview.XSize, bOverview.YSize)  
        
    return np.stack((rOverview ,gOverview ,bOverview),axis=2)


def imread_gdal_mask(data_name,level):
    
    level=level-1

    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        
    rOverview = rOverview.ReadAsArray(0,0, rOverview.XSize, rOverview.YSize) 
    
    return rOverview









for data_type in ['test']:

    model_spec='s1_dense_net_preatrained_0.9964523900000001__020000.pkl'
    model_name='s1_dense_net_preatrained'
    
    data_path='E:/histolky_drazdany_data/lungs/dataset/'+ data_type  +'/data' 
    cn='no'
    scale='1'
#    dense_down=2#0
    dense_down=4
    
    write_scale=1#wrtie smaler
    down_scale=16#network scale
    
    fg_scale_down=8#fg->img scale
    
#    step=320
#    border=0
    
    step=246*4
    border=106
    
    lvl=0

    
#    auc_sampling_per=0.001*(down_scale)**2
    auc_sampling_per=1
    
    if auc_sampling_per>1:
        auc_sampling_per=1
    
    
    
    model_path='../results/' +model_name+ '/models/' +model_spec
    save_folder='../results/' +model_name  +'/' +data_type
    
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
#    file_names=file_names[30:]
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
            
           
            augmenters=[]

            augmenters.append(augmenter.ToFloat())
            augmenters.append(augmenter.RangeToRange((0,255),(0,1)))
        
                
            augmenters.append(augmenter.RangeToRange((0,1),(-1,1)))
            augmenters.append(augmenter.TorchFormat())

        
            imgs,_=augmenter.augment_all(augmenters,imgs,[])
            
            
            
            img=imgs[0]
            
            

            img=img.view(1,np.shape(img)[0],np.shape(img)[1],np.shape(img)[2]).cuda()
            
            
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
        
        
        lvl_tmp=4
        img=imread_gdal(img_filename,lvl_tmp)
        if 'tumor' in lbl_name:
            lbl=imread_gdal_mask(lbl_name,lvl_tmp)>0
        else:
            lbl=np.zeros(np.shape(img)[:-1])>0
        res=imread_gdal_mask(save_name,0)
        
        x=np.min((np.shape(img)[0],np.shape(res)[0]))
        y=np.min((np.shape(img)[1],np.shape(res)[1]))
        
        img=img[:x,:y,:]
        res=res[:x,:y]
        
        
        img=img_with_cont(img,lbl,(255,0,0),2)
        green=np.stack((np.zeros(np.shape(res)),np.ones(np.shape(res)),np.zeros(np.shape(res))),axis=2)
        composite=img+green*np.repeat(np.expand_dims(res*255,axis=2),3,axis=2)
        composite[composite>255]=255
        composite=np.round(composite).astype(np.uint8)
        
        res=img_with_cont(np.repeat(np.expand_dims(res*255,axis=2),3,axis=2),lbl,(255,0,0),2)
        
        cotrol_name_res=save_name +'kontola1.tiff'
        cotrol_name_img=save_name +'kontola2.tiff'
        cotrol_name_trans=save_name +'kontola3.tiff'
        imsave(cotrol_name_res,res)
        imsave(cotrol_name_img,img)
        imsave(cotrol_name_trans,composite)
        
       
    
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








