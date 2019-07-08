import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import cv2
import gdal
from skimage.io import imread


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



data_type='test'
model_name='s1_dense_net_preatrained'
data_path='E:/histolky_drazdany_data/lungs/dataset/'+ data_type  +'/data' 
results_path=save_folder='../results/' +model_name  +'/' +data_type



file_names=[]
for root, dirs, files in os.walk(data_path):
    for name in files:
        if name.endswith(".tif"):
            file_names.append(root+'/'+name)

for_auc_res=[]
for_auc_gt=[]
TP=np.array(0,dtype=np.float64)
TN=np.array(0,dtype=np.float64)
FP=np.array(0,dtype=np.float64)
FN=np.array(0,dtype=np.float64)

i=-1
for img_filename in file_names:
    i+=1
    print(i,img_filename)
    
    fg_name=img_filename
    fg_name=fg_name.replace('/data/','/fg/')
    
    lbl_name=fg_name
    lbl_name=lbl_name.replace('/fg/','/mask/')
#    lbl_name=lbl_name.replace('/fg/','/mask_dil_lvl2_32/')
    lbl_name=lbl_name.replace('.tif','_mask.tif')
    
    tmp=img_filename.split('/')
    tmp=tmp[-1]
    result_name=results_path +'/'+ tmp
    
    
    lvl=4
    auc_sampling_per=1
    
    img=imread_gdal_mask(result_name,0)
    data=imread_gdal(img_filename,lvl)
    if '_tumor' in lbl_name:
        lbl=imread_gdal_mask(lbl_name,lvl)
#        lbl=imread_gdal_mask(lbl_name,lvl-2)
    else:
        lbl=np.zeros(np.shape(img))
    fg=imread(fg_name)
    fg=cv2.resize(fg,None,fx=1/2, fy=1/2, interpolation = cv2.INTER_NEAREST)>0
    
    
    
    minx=np.min((fg.shape[0],lbl.shape[0],img.shape[0]))
    miny=np.min((fg.shape[1],lbl.shape[1],img.shape[1]))
    img=img[:minx,:miny]
    fg=fg[:minx,:miny]
    lbl=lbl[:minx,:miny]
    
    
    fg[lbl==1]=0
    lbl=lbl>1
    
        
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




print('TP ' + str(TP))
print('TN ' + str(TN))
print('FP ' + str(FP))
print('FN ' + str(FN))
print('auc ' + str(auc_s))
print('sen ' + str(sen))
print('spe ' + str(spe))
print('acc ' + str(acc))
print('dice ' + str(dice))