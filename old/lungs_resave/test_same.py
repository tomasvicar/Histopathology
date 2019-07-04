
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.color import rgb2lab
from skimage.io import imsave
import os
import gdal
import matplotlib.pyplot as plt

def imread_gdal(data_name,level):
    
  

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

p=r'D:\MPI_CBG\data_plice\data\normal_tif'

file_names=[]
for folder_path in p:
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if name.endswith("normal.tif"):
                file_names.append(root+'\\'+name)
                
                
                
for file_name in file_names:
    i+=1
    
    print(str(i) + '/'+ str(len(file_names)) + '  ' + file_name)
    

    
    img_s=imread_gdal(file_name,5)
    
    plt.figure()
    plt.imshow(img_s)
    plt.title(file_name)
    
    