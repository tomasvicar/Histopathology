import os
import gdal
import cv2
import matplotlib.pyplot as plt
import numpy as np


def img_with_cont(img,mask,c=(0,0,255)):
    
    mask=np.uint8((np.squeeze(mask)>0)*255)
    mask=np.squeeze(mask)
    
    
    img=np.array(img,dtype=np.uint8)
    
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, c, 10)
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





lvl=5
data_type='test'

model_name='1s_no_unet'
save_folder='..\\results\\' +model_name +'\\'+data_type
data_path=r'Y:\CELL_MUNI\histolky\dataset\\'+  data_type +'_cam\data'

file_names=[]
for root, dirs, files in os.walk(data_path):
    for name in files:
        if name.endswith(".tif"):
            file_names.append(root+'\\'+name)
            
i=36
file_names=file_names[i:i+1]


for img_filename in file_names:

    
    fg_name=img_filename
    fg_name=fg_name.replace('\\data\\','\\fg\\')
    
    lbl_name=fg_name
    lbl_name=lbl_name.replace('\\fg\\','\\mask\\')
    lbl_name=lbl_name.replace('.tif','_mask.tif')
    
    tmp=img_filename.split('\\')
    tmp=tmp[-1]
    result_name=save_folder +'\\'+ tmp
    
    
        
    result=imread_gdal_mask(result_name,lvl-1)
    data=imread_gdal(img_filename,lvl+1)
    if '_tumor' in lbl_name:
        gt=imread_gdal_mask(lbl_name,lvl+1)
    else:
        gt=np.zeros(np.shape(result))
    
    
#    plt.imshow(result>0.9)
    
    
    img_c=img_with_cont(data,result>0.90,c=(0,0,255));
    img_c=img_with_cont(img_c,gt>0,c=(255,0,0));
    plt.imshow(img_c)
        
    plt.figure()
    plt.imshow(result)
        
    
    
    
    
    
    
    