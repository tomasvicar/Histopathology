import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.color import rgb2lab
from skimage.io import imsave
import os
import gdal
import gc
import matplotlib.pyplot as plt
from skimage.io import imsave
from skimage.io import imread
import time
from scipy.ndimage.morphology import binary_dilation

from scipy.ndimage import generate_binary_structure


def imread_gdal_mask(data_name,level):
    
    level=level-1

    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        
    rOverview = rOverview.ReadAsArray(0,0, rOverview.XSize, rOverview.YSize) 
    
    return rOverview

def write_gdal_mask(data,name):
    
    dst_ds = gdal.GetDriverByName('GTiff').Create(name, np.shape(data)[1], np.shape(data)[0], 1, gdal.GDT_Byte,['COMPRESS=LZW','BIGTIFF=YES','TILED=YES'])
    
    if len(np.shape(data))>2:
        kk=np.shape(data)[2]
    else:
        kk=1
    for k in range(kk):
        if len(np.shape(data))>2:
            dst_ds.GetRasterBand(k+1).WriteArray(data[:,:,k])   # write r-band to the raster
        else:
            dst_ds.GetRasterBand(k+1).WriteArray(data)   # write r-band to the raster

    dst_ds.BuildOverviews("NEAREST", [2,4,8,16,32,64])
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None



folder_paths=[r'D:\MPI_CBG\camelyon16\dataset\train\data',r'D:\MPI_CBG\camelyon16\dataset\test\data',r'D:\MPI_CBG\camelyon16\dataset\valid\data']
lvl=3

for folder_path in folder_paths:
    
    tmp = folder_path.split('\\')
    test_train_valid=tmp[-2]

    file_names_tumor=[]
    file_names_normal=[]
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if name.endswith(".tif") and name.find("tumor")>=0:
                file_names_tumor.append(root+'\\'+name)   
            elif name.endswith(".tif") and name.find("normal")>=0:
                file_names_normal.append(root+'\\'+name)
            else:
                raise('chyba')
                
                
                
    for file_name in file_names_tumor:
        
        mask_name=file_name.replace('\\data\\','\\mask\\')
        
        mask_name=mask_name[:-4] + '_mask.tif'
        
        dil_mask_name=mask_name.replace('\\mask\\','\\mask_dil_lvl3\\')
        
        
        mask0=imread_gdal_mask(mask_name,lvl)>0
        
#        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(16,16))
        s=np.ones((16,16))>0
        mask=binary_dilation(mask0,s)
        
        write_gdal_mask(mask,dil_mask_name)        
        
        
        
        
        
                
    