import numpy as np
from patcher_test import Patcher 
import torch
from torch.autograd import Variable
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import augmenter
from skimage.io import imsave
from utils.utils import img_with_cont
from utils.gdal_fcns import imread_gdal,getsize_gdal
import pixel_net_eq256_kplus as pixel_net


#results_path='/media/ubmi/DATA1/vicar/code/results/s1_pixel_baseinfsampler'
#model='0.9775__00347700.pkl'


results_path='/media/ubmi/DATA1/vicar/code/results/s2_pixel_baseinfsampler'
model='0.801__00025200.pkl'


data_path='/media/ubmi/DATA2/vicar/cam_dataset/test/data' 
#data_path='/media/ubmi/DATA2/vicar/cam_dataset/train/data' 



lvl=2###based on network input scale
fg_lvl=4 ##vased on fg mask scale

dense_down=4
fg_lbl_lvl_get=6
write_lvl=6

step=256*8
border=120 #(net_orig_input_size-2**num_of_pools)/2

lvl_control=6




model_path=results_path+ os.sep + model
save_folder=results_path +os.sep+'results'

try:
    os.makedirs(save_folder)
except:
    print('folder existuje')



     
    

file_names=[]
for root, dirs, files in os.walk(data_path):
    for name in files:
        if name.endswith(".tif"):
            file_names.append(root+os.sep+name)
    
      
file_names=file_names[13:]
#file_names = [file_names[i] for i in [1,6]]

model= torch.load(model_path)
model=model.cuda()
model.dense_on(dense_down)
model.eval()




for ii,img_filename in enumerate(file_names):
    print(str(ii) +os.sep+ str(len(file_names)))

    fg_name=img_filename
    fg_name=fg_name.replace(os.sep+'data'+os.sep,os.sep+'fg'+os.sep)
    
    lbl_name=fg_name
    lbl_name=lbl_name.replace(os.sep+'fg'+os.sep,os.sep+'mask'+os.sep)
    lbl_name=lbl_name.replace('.tif','_mask.tif')
    
    tmp=img_filename.split(os.sep)
    tmp=tmp[-1]
    save_name=save_folder +os.sep+ tmp
    

    
    patcher=Patcher(img_filename,fg_name,lbl_name,save_name,lvl=lvl,fg_lvl=fg_lvl,
                    fg_lbl_lvl_get=fg_lbl_lvl_get,step=step,border=border,write_lvl=write_lvl,
                    lbl_exist='tumor' in lbl_name)
    
    
    img_sizes=[]

    TP=np.array(0,dtype=np.float64)
    TN=np.array(0,dtype=np.float64)
    FP=np.array(0,dtype=np.float64)
    FN=np.array(0,dtype=np.float64)
    
    
    img_size=0
    for i in range(patcher.get_num_of_patches()):
        print(str(i) +'/'+ str(patcher.get_num_of_patches()))

        img0,fg,lbl=patcher.read()
        
        plt.imshow(img0[:,:,0])
        plt.show()
        
        
        imgs=[img0]
        
        augmenters=[]
        augmenters.append(augmenter.ToFloat())
        augmenters.append(augmenter.RangeToRange((0,255),(-1,1)))
        augmenters.append(augmenter.TorchFormat())
        imgs,_=augmenter.augment_all(augmenters,imgs,[])
        
        img=imgs[0]
        

        img=img.view(1,np.shape(img)[0],np.shape(img)[1],np.shape(img)[2]).cuda()
        
        
        img=model(img)
        img=torch.sigmoid(img)
        img=img[0,0,:,:].data.cpu().numpy()

        plt.imshow(np.concatenate((fg,img*255,lbl),axis=1))
        plt.show()

        
        img[fg==0]=0
        
        TP+=np.sum(np.bitwise_and(np.bitwise_and(lbl>0,img>0.5),fg>0))
        TN+=np.sum(np.bitwise_and(np.bitwise_and(lbl==0,img<=0.5),fg>0))
        FP+=np.sum(np.bitwise_and(np.bitwise_and(lbl==0,img>0.5),fg>0))
        FN+=np.sum(np.bitwise_and(np.bitwise_and(lbl>0,img<=0.5),fg>0))
        

        img_size=img_size+np.sum(fg>0)
        


        patcher.write(img*255)
        
    img_sizes.append(img_size)
    
    
    sen=TP/(TP+FN)
    spe=TN/(TN+FP)
    acc=(TP+TN)/(TP+FP+FN+TN)
    dice=2*TP/(2*TP+FP+FN)
    
    print('TP ' + str(TP))
    print('TN ' + str(TN))
    print('FP ' + str(FP))
    print('FN ' + str(FN))
    print('sen ' + str(sen))
    print('spe ' + str(spe))
    print('acc ' + str(acc))
    print('dice ' + str(dice))
    
    
    
    
    
    img=imread_gdal(img_filename,lvl_control)
    if 'tumor' in lbl_name:
        lbl=imread_gdal(lbl_name,lvl_control)>0
    else:
        lbl=np.zeros(np.shape(img)[:-1])>0
    res=imread_gdal(save_name,lvl_control-write_lvl,dtype=np.float32)
    x=np.min((np.shape(img)[0],np.shape(res)[0]))
    y=np.min((np.shape(img)[1],np.shape(res)[1]))
       
    img=img[:x,:y,:]
    res=res[:x,:y]
    
    
    img=img_with_cont(img,lbl,(255,0,0),2)
    green=np.stack((np.zeros(np.shape(res)),np.ones(np.shape(res)),np.zeros(np.shape(res))),axis=2)
    composite=img+green*np.repeat(np.expand_dims(res,axis=2),3,axis=2)
    composite[composite>255]=255
    composite=np.round(composite).astype(np.uint8)
    
    res=img_with_cont(np.repeat(np.expand_dims(res,axis=2),3,axis=2),lbl,(255,0,0),2)
    
    cotrol_name_res=save_name +'_control1.tiff'
    cotrol_name_img=save_name +'_control2.tiff'
    cotrol_name_trans=save_name +'_control3.tiff'
    imsave(cotrol_name_res,res)
    imsave(cotrol_name_img,img)
    imsave(cotrol_name_trans,composite)
    
   
