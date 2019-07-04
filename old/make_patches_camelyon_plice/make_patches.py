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
    
    
    patch_size_s=96
    patch_size_l=96*2
    patch_size_U=320
    
    if data_name.find("normal")>=0:
        pouzit_masku=0
        
    elif data_name.find("tumor")>=0:
        pouzit_masku=1
    
    
        
    fg=imread(fg_name)
    if pouzit_masku:
        lbl=imread_gdal_mask(mask_name,lvl+3)
    else:
        lbl=np.zeros(np.shape(fg), dtype=np.bool)
    
    fg=fg[:np.min([np.shape(lbl)[0],np.shape(fg)[0]]),:np.min([np.shape(lbl)[1],np.shape(fg)[1]])]
        
    lbl=lbl[:np.min([np.shape(lbl)[0],np.shape(fg)[0]]),:np.min([np.shape(lbl)[1],np.shape(fg)[1]])]
    
#    fg=cv2.resize(fg,None,fx=4, fy=4, interpolation = cv2.INTER_NEAREST)>0
    
    
        
    
        
#    print(np.shape(fg))
#    print(np.shape(lbl))
    if tumor:
        tmp=lbl>0
    else:
        tmp=np.bitwise_and(fg>0,lbl==0)
        
    if dataset=='plice':
        tmp[lbl==1]=0
        
    

        
            
    tmp=clear_border(tmp,4*(1+int(np.ceil(patch_size_U/2/2+1))))
    pozice=np.where(tmp>0)
    
    
    rr=np.random.choice(len(pozice[0]),zaras)
    
    
    
    i=-1
    patch_ss=[]
    patch_s=[]
    patch_l=[]
    patch_U=[]
    mask_ss=[]
    mask_s=[]
    mask_l=[]
    mask_U=[]
    for r in rr:
        i+=1
        

        pozicex=pozice[1][r]*2*4#x4x2 is camelyon
        pozicey=pozice[0][r]*2*4
        pozicex+=np.random.randint(2*4)
        pozicey+=np.random.randint(2*4)
        
#        print('dfdf' +str( lbl[int(pozicex/2),int(pozicey/2)]))
        
        if dataset=='cam':
            if pouzit_masku:
                mask_ss.append(get_patches_mask_gdal(mask_name,[pozicex*2,pozicey*2],patch_size_s,lvl-1))
            else:
                mask_ss.append(np.zeros((patch_size_s,patch_size_s)))
            patch_ss.append(get_patches_data_gdal(data_name,[pozicex*2,pozicey*2],patch_size_s,lvl-1))
    
        if pouzit_masku:
            mask_s.append(get_patches_mask_gdal(mask_name,[pozicex,pozicey],patch_size_s,lvl))
        else:
            mask_s.append(np.zeros((patch_size_s,patch_size_s)))
        patch_s.append(get_patches_data_gdal(data_name,[pozicex,pozicey],patch_size_s,lvl))
        
        if pouzit_masku:
            mask_l.append(get_patches_mask_gdal(mask_name,[pozicex,pozicey],patch_size_l,lvl))
        else:
            mask_l.append(np.zeros((patch_size_l,patch_size_l)))
        patch_l.append(get_patches_data_gdal(data_name,[pozicex,pozicey],patch_size_l,lvl))
        
        if pouzit_masku:
            mask_U.append(get_patches_mask_gdal(mask_name,[pozicex,pozicey],patch_size_U,lvl))
        else:
            mask_U.append(np.zeros((patch_size_U,patch_size_U)))
        patch_U.append(get_patches_data_gdal(data_name,[pozicex,pozicey],patch_size_U,lvl))        
        
    if dataset=='plice':
        return np.stack(patch_s,axis=3),np.stack(patch_l,axis=3),np.stack(patch_U,axis=3),np.stack(mask_s,axis=2),np.stack(mask_l,axis=2),np.stack(mask_U,axis=2)
    else:
        return np.stack(patch_ss,axis=3),np.stack(patch_s,axis=3),np.stack(patch_l,axis=3),np.stack(patch_U,axis=3),np.stack(mask_ss,axis=2),np.stack(mask_ss,axis=2),np.stack(mask_l,axis=2),np.stack(mask_U,axis=2)







dataset='plice'
patch_path=r'D:\MPI_CBG\data_plice\patches'    
folder_paths=[r'D:\MPI_CBG\data_plice\dataset\train\data',r'D:\MPI_CBG\data_plice\dataset\test\data',r'D:\MPI_CBG\data_plice\dataset\valid\data']
#folder_paths=[r'D:\MPI_CBG\data_plice\dataset\valid\data']
lvl=0

#dataset='cam'
#patch_path=r'D:\MPI_CBG\camelyon16\patches'    
##folder_paths=[r'D:\MPI_CBG\camelyon16\dataset\train\data',r'D:\MPI_CBG\camelyon16\dataset\test\data',r'D:\MPI_CBG\camelyon16\dataset\valid\data']
#folder_paths=[r'D:\MPI_CBG\camelyon16\dataset\test\data',r'D:\MPI_CBG\camelyon16\dataset\valid\data']
#lvl=1


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
            kolik=60000   
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
        
#        if test_train_valid=='test':
#            fg_name=fg_name.replace('tumor_','')
#            fg_name=fg_name.replace('normal_','')
        
        mask_name=file_name.replace('\\data\\','\\mask\\')
        mask_name=mask_name[:-4]+'_mask.tif'
        
        if dataset=='plice':
            patch_s,patch_l,patch_U,mask_s,mask_l,mask_U=get_patches(file_name,fg_name,mask_name,zaras,lvl,dataset,tumor)
        else:
            patch_ss,patch_s,patch_l,patch_U,mask_ss,mask_s,mask_l,mask_U=get_patches(file_name,fg_name,mask_name,zaras,lvl,dataset,tumor)
        
        patch_l=patch_l[::2,::2,:,:]
        mask_l=mask_l[::2,::2,:]
        
        lbl=int(tumor)
        
        tmp = file_name.split('\\')
        train_test_valid=tmp[-3]
        name=tmp[-1][:-4]
        
        for kk in range(zaras):
            
            time_name='{:.0f}'.format(np.round(time.time()*1e+10))
            save_path_folder=patch_path + '\\' + train_test_valid + '\\' + name + '\\'
            
            if dataset=='cam':
                imsave(save_path_folder+'data_ss'+'_'  + str(lbl) + time_name +'.tif',patch_ss[:,:,:,kk],compress=6)
                imsave(save_path_folder+'mask_ss' +'_' + str(lbl) + time_name +'.tif',mask_ss[:,:,kk],compress=6)
            
            imsave(save_path_folder+'data_s'+'_'  + str(lbl) + time_name +'.tif',patch_s[:,:,:,kk],compress=6)
            imsave(save_path_folder+'data_l' +'_' + str(lbl) + time_name +'.tif',patch_l[:,:,:,kk],compress=6)
            imsave(save_path_folder+'data_U' +'_' + str(lbl) + time_name +'.tif',patch_U[:,:,:,kk],compress=6)
            imsave(save_path_folder+'mask_s' +'_' + str(lbl) + time_name +'.tif',mask_s[:,:,kk],compress=6)
            imsave(save_path_folder+'mask_l' +'_' + str(lbl) + time_name +'.tif',mask_l[:,:,kk],compress=6)
            imsave(save_path_folder+'mask_U' +'_' + str(lbl) + time_name +'.tif',mask_U[:,:,kk],compress=6)
        
        
          
    