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


def clear_border(maskin,n):
    maskout=maskin
    maskout[0:n,:]=False
    maskout[-n:,:]=False
    maskout[:,0:n]=False
    maskout[:,-n:]=False
    return maskout



def imread_gdal_mask(data_name,level):
    
    level=level-1

    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        
    rOverview = rOverview.ReadAsArray(0,0, rOverview.XSize, rOverview.YSize) 
    
    return rOverview


def get_patches_data_gdal(data_name,pos,patch_size,level):
    
    level=level-1
    patch_size_half=patch_size/2
    

    gdalObj = gdal.Open(data_name)
    
#     pos=pos[(1,0),:]

    
    nBands = gdalObj.RasterCount
    rOverview = gdalObj.GetRasterBand (1)
    gOverview = gdalObj.GetRasterBand (2)
    bOverview = gdalObj.GetRasterBand (3)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        gOverview = gOverview.GetOverview(level)
        bOverview = bOverview.GetOverview(level)
    
#     width = rOverview.XSize
#     height = rOverview.YSize


        

    
    overview = np.zeros((patch_size, patch_size, nBands), dtype=np.uint8)
    overview[:,:,0] = rOverview.ReadAsArray(int(pos[0]-patch_size_half), int(pos[1]-patch_size_half), int(patch_size), int(patch_size)) 
    overview[:,:,1] = gOverview.ReadAsArray(int(pos[0]-patch_size_half), int(pos[1]-patch_size_half), int(patch_size), int(patch_size)) 
    overview[:,:,2] = bOverview.ReadAsArray(int(pos[0]-patch_size_half), int(pos[1]-patch_size_half), int(patch_size), int(patch_size)) 
        
    return overview



def get_patches_mask_gdal(data_name,pos,patch_size,level):
    
    level=level-1
    
    patch_size_half=patch_size/2

    gdalObj = gdal.Open(data_name)


    
    nBands = gdalObj.RasterCount
    rOverview  = gdalObj.GetRasterBand(1)

    if level>=0:
        rOverview = rOverview.GetOverview(level)
    

    
    overview = np.zeros((patch_size,patch_size), dtype=np.uint8)
    tmp = rOverview.ReadAsArray(int(pos[0]-patch_size_half), int(pos[1]-patch_size_half), int(patch_size), int(patch_size)) 
    overview[:,:]=tmp
    
#    print(rOverview.XSize, rOverview.YSize,pos)
    
#    print(overview[int(patch_size_half),int(patch_size_half)])
    return overview



def get_patches(data_name,fg_name,mask_name,zaras,lvl,dataset,tumor):
    
    
    patch_size=512

    if data_name.find("normal")>=0:
        pouzit_masku=0
        
    elif data_name.find("tumor")>=0:
        pouzit_masku=1
    
    
        
    fg=imread(fg_name)
    if pouzit_masku:
        lbl=imread_gdal_mask(mask_name,lvl+3)#lvl1 ->+1
    else:
        lbl=np.zeros([int(np.shape(fg)[0]*2),int(np.shape(fg)[1]*2)], dtype=np.bool)#####plice
    
    fg=cv2.resize(fg,None,fx=2, fy=2, interpolation = cv2.INTER_NEAREST)>0#####plice
    
    
    fg=fg[:np.min([np.shape(lbl)[0],np.shape(fg)[0]]),:np.min([np.shape(lbl)[1],np.shape(fg)[1]])]
        
    lbl=lbl[:np.min([np.shape(lbl)[0],np.shape(fg)[0]]),:np.min([np.shape(lbl)[1],np.shape(fg)[1]])]
    
#    
    
    
    if tumor:
        tmp=lbl>0
    else:
        tmp=np.bitwise_and(fg>0,lbl==0)
        
    if dataset=='plice':
        tmp[lbl==1]=0
        
    

        
            
    tmp=clear_border(tmp,2*(1+int(np.ceil(patch_size/2/2+1))))
    pozice=np.where(tmp>0)

    
    rr=np.random.choice(len(pozice[0]),zaras)
        
    
    
    i=-1
    patch=[]
    mask=[]

    for r in rr:
        i+=1
        

        pozicex=pozice[1][r]*2*2*2
        pozicey=pozice[0][r]*2*2*2
        pozicex+=np.random.randint(2*2*2)
        pozicey+=np.random.randint(2*2*2)
        
        

        if pouzit_masku:
            mask.append(get_patches_mask_gdal(mask_name,[pozicex,pozicey],patch_size,lvl))
        else:
            mask.append(np.zeros((patch_size,patch_size)))
        patch.append(get_patches_data_gdal(data_name,[pozicex,pozicey],patch_size,lvl))
        
        
        if pouzit_masku:
            mask.append(get_patches_mask_gdal(mask_name,[pozicex,pozicey],patch_size,lvl-1))
        else:
            mask.append(np.zeros((patch_size,patch_size)))
        patch.append(get_patches_data_gdal(data_name,[pozicex,pozicey],patch_size,lvl-1))
        
        
    
    return patch,mask





#dataset='plice'
#patch_path=r'D:\MPI_CBG\data_plice\patches'    
#folder_paths=[r'D:\MPI_CBG\data_plice\dataset\train\data',r'D:\MPI_CBG\data_plice\dataset\test\data',r'D:\MPI_CBG\data_plice\dataset\valid\data']
##folder_paths=[r'D:\MPI_CBG\data_plice\dataset\valid\data']
#lvl=0

dataset='cam'
patch_path=r'D:\MPI_CBG\camelyon16\patches_0s'    
folder_paths=[r'D:\MPI_CBG\camelyon16\dataset\train\data',r'D:\MPI_CBG\camelyon16\dataset\test\data',r'D:\MPI_CBG\camelyon16\dataset\valid\data']
#folder_paths=[r'D:\MPI_CBG\camelyon16\dataset\valid\data']
lvl=0


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
                
    if dataset=="plice":
        if test_train_valid=='test':
            kolik=2000
        if test_train_valid=='valid':  
            kolik=2000
        if test_train_valid=='train':  
            kolik=30000  
            
    elif dataset=="cam":
        if test_train_valid=='test':
            kolik=2000
        if test_train_valid=='valid':  
            kolik=2000
        if test_train_valid=='train':  
            kolik=50000   
    else:
        raise('chyba')
            
                
    zaras=5
    
    for k in range(0,kolik,zaras):
        
        
        
        tumor=k>kolik/2

            
        if tumor:
            file_names=file_names_tumor
        else:
            file_names=file_names_tumor+file_names_normal
            
            
        file_name=file_names[np.random.choice(len(file_names),1)[0]]   
        
        
        print(test_train_valid + '   ' + str(k)  + '   ' + file_name)
        
        fg_name=file_name.replace('\\data\\','\\fg\\')
        
        
        mask_name=file_name.replace('\\data\\','\\mask\\')
        mask_name=mask_name[:-4]+'_mask.tif'
        
        
        patch,mask=get_patches(file_name,fg_name,mask_name,zaras,lvl,dataset,tumor)
        
        
       
        
        lbl=int(tumor)
        
        tmp = file_name.split('\\')
        train_test_valid=tmp[-3]
        name=tmp[-1][:-4]
        
        for kk in range(zaras):
            
            time_name='{:.0f}'.format(np.round(time.time()*1e+10))
            save_path_folder=patch_path + '\\' + train_test_valid + '\\' + name + '\\'
            
            try:
                os.makedirs(save_path_folder)
            except:
                pass
            
            patch_tmp=patch[kk]
            mask_tmp=mask[kk]
            midle=int(np.shape(patch_tmp)[0]/2)
            i=[int(96/2),int(208/2),int(208*2/2)]
            
            for k in range(len(i)):
                p=patch_tmp[midle-i[k]:midle+i[k],midle-i[k]:midle+i[k],:]
                m=mask_tmp[midle-i[k]:midle+i[k],midle-i[k]:midle+i[k]]
                
                
                
                imsave(save_path_folder+'data_'+ str(k) +'_' + str(lbl) + time_name +'.tif',p,compress=6)
    
                imsave(save_path_folder+'mask_'+ str(k) +'_' + str(lbl) + time_name +'.tif',m,compress=6)
        
        
          
    