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


def img_with_cont(img,mask,color=(0,0,255)):
    
    mask=np.uint8((np.squeeze(mask)>0)*255)
    mask=np.squeeze(mask)
    
    
    img=np.array(img,dtype=np.uint8)
    
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, color, 10)
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


lvl=7   
folder_paths=[r'D:\MPI_CBG\camelyon16\dataset\train\data',r'D:\MPI_CBG\camelyon16\dataset\test\data',r'D:\MPI_CBG\camelyon16\dataset\valid\data']
folder_paths=[r'D:\MPI_CBG\camelyon16\dataset\test\data']


#lvl=3
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
#    ########xremove
#    if file_name!=r'D:\MPI_CBG\data_plice\dataset\test\data\2017_11_30__0034-4.czi.labeling_tumor.tif':
#        continue  
#    ##########
    
    print(str(i) + '/'+ str(len(file_names)) + '  ' + file_name)
    
    mask_save_name=file_name.replace('\\data\\','\\fg\\')
    
    lbl_save_name=file_name.replace('\\data\\','\\mask\\')
    
    lbl_load_name=file_name.replace('\\data\\','\\mask\\')
    lbl_load_name=lbl_load_name.replace('.tif','_mask.tif')
    
    
    img_s=imread_gdal(file_name,lvl)
    
    
#    fg=get_fg(img_s)
#    
#    imsave(mask_save_name,fg,compress=6)
#
#    img_fg_cont=img_with_cont(img_s,fg)
#    
#    
#    imsave(mask_save_name+'kontrola.tif',img_fg_cont,compress=6)
    
    
    
    
    if 'tumor' in lbl_load_name:
        lbl=imread_gdal_mask(lbl_load_name,lvl)
    else:
        lbl=np.zeros((np.shape(img_s)[0],np.shape(img_s)[1]))
    
    img_lbl_cont=img_with_cont(img_s,lbl>0,(0,0,255))
    img_lbl_cont=img_with_cont(img_lbl_cont,lbl==2,(0,255,0))
    img_lbl_cont=img_with_cont(img_lbl_cont,lbl==3,(255,0,0))
    
    
    print(np.sum(lbl==1))
    print(np.sum(lbl==2))
    print(np.sum(lbl==3))
    
    imsave(lbl_save_name+'kontrola.tif',img_lbl_cont,compress=6)
    
#    plt.imshow(img_s)
#    
#    
#    a=fdafdf









