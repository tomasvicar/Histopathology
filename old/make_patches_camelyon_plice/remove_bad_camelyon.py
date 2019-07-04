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


def img_with_cont(img,mask):
    
    mask=np.uint8((np.squeeze(mask)>0)*255)
    mask=np.squeeze(mask)
    
    
    img=np.array(img,dtype=np.uint8)
    
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0,0,255), 10)
    return img



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



def apply_hysteresis_threshold(image, low, high):
    from scipy import ndimage as ndi
    
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = image > low
    mask_high = image > high
    # Connected components of mask_low
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded


def get_fg(img):

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    lab = cv2.GaussianBlur(lab,(11,11), 0)
    
    fg=np.bitwise_or(np.sum(img==0,axis=2)==3,np.sum(img==255,axis=2)==3)==0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    fg = cv2.erode(fg.astype(np.uint8), kernel)==1
    
    
    for_otsu=lab[:,:,1]
    for_otsu=for_otsu[fg]
    threshold = threshold_otsu(for_otsu)
##    print(threshold)
    
#    threshold=135

    
    l=lab[:,:,1]
    
    del lab
    del img
#    del for_otsu
    
    gc.collect()
    gc.collect()
    


    mask = (apply_hysteresis_threshold(l, threshold*0.96, threshold)*fg).astype(np.uint8)*255
    
    
#      
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
    mask = cv2.erode(mask, kernel)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
    mask = cv2.dilate(mask, kernel)
    
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#    mask = cv2.dilate(mask, kernel)
#    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#    mask = cv2.erode(mask, kernel)
    
    
    return mask


lvl=3    
folder_paths=[r'D:\MPI_CBG\camelyon16\dataset\train\data']
#lvl=2
#folder_paths=[r'D:\MPI_CBG\data_plice\dataset\train\data',r'D:\MPI_CBG\data_plice\dataset\test\data',r'D:\MPI_CBG\data_plice\dataset\valid\data']
file_names=[]
for folder_path in folder_paths:
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if name.endswith(".tif"):
                file_names.append(root+'\\'+name)
                
#file_names=file_names[0:1]                
                
i=0                
for file_name in file_names:
    i+=1
    
    
    tmp=file_name.split('\\')
    tmp=tmp[-1]
    
    if tmp=='tumor_018.tif' or tmp=='tumor_025.tif' or tmp=='tumor_046.tif' or tmp=='tumor_051.tif' or tmp=='tumor_054.tif' or tmp=='tumor_067.tif' or tmp=='tumor_079.tif' or tmp=='tumor_092.tif' or tmp=='tumor_095.tif':
        pass
    else:
        continue
    
    
    print(str(i) + '/'+ str(len(file_names)) + '  ' + file_name)
    
    mask_save_name=file_name.replace('\\data\\','\\fg\\')
    
    img_s=imread_gdal(file_name,lvl)
    
    
    fg=get_fg(img_s)
    
    
    s1=np.shape(fg)[0]
    s0=np.shape(fg)[1]
    
    if tmp=='tumor_018.tif':
        fg[:int(np.round(s1/2)),:]=0
    
    if tmp=='tumor_025.tif':
        fg[:int(np.round(s1*2344/3584)),:]=0
    
    if tmp=='tumor_046.tif':
        fg[int(np.round(s1/2)):,:]=0
        
    if tmp=='tumor_051.tif':
        fg[:,int(np.round(s0/2)):]=0    
        
        
    if tmp=='tumor_054.tif':
        fg[:int(np.round(s1*4300/7168)),]=0  
        
    if tmp=='tumor_067.tif':
        fg[:int(np.round(s1/2)),]=0  
        
        
    if tmp=='tumor_079.tif':
        fg[:int(np.round(s1*2541/3584)),]=0 
        
    if tmp=='tumor_092.tif':
        fg[:,int(np.round(s0*1200/7168)):]=0 
        
    if tmp=='tumor_095.tif':
        fg[:,int(np.round(s0*2673/5632)):]=0 
        
        
    
    imsave(mask_save_name,fg,compress=6)

    img_s=img_with_cont(img_s,fg)
    
    
    imsave(mask_save_name+'kontrola.tif',img_s,compress=6)
    
#    plt.imshow(img_s)
#    
#    
#    a=fdafdf









