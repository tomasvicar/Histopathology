import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.color import rgb2lab
from skimage.io import imread
import os
import gdal
from get_stain_macenko_vah_fun import get_stain_macenko_vah
import gc

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


def get_m_std(img_s,fg):
    
    fg=fg[:np.min([np.shape(img_s)[0],np.shape(fg)[0]]),:np.min([np.shape(img_s)[1],np.shape(fg)[1]])]
        
    img_s=img_s[:np.min([np.shape(img_s)[0],np.shape(fg)[0]]),:np.min([np.shape(img_s)[1],np.shape(fg)[1]]),:]
    
    m=[]
    std=[]
    for k in range(3):
        c=img_s[:,:,k]
        c=c[fg>0]
        m.append(np.mean(c))
        std.append(np.std(c))
        
    rgb_m,rgb_std=m,std
    
    img_s=np.reshape(img_s,(-1,3))
    img_s=img_s[np.nonzero(fg.flatten())[0],:]
    
    kolik=100000     
    r=np.random.choice(np.shape(img_s)[0],kolik)
    img_s=img_s[r,:]
        
    img_s=np.reshape(img_s,(-1,1,3))
    
    
    del c
    gc.collect()
    
    img_s=rgb2lab(img_s)
    m=[]
    std=[]
    for k in range(3):
        c=img_s[:,:,k]
#        c=c[fg>0]
        m.append(np.mean(c))
        std.append(np.std(c))
        
    lab_m,lab_std=m,std
    
    
    return rgb_m,rgb_std,lab_m,lab_std






#dataset='plice'
#patch_path=r'D:\MPI_CBG\data_plice\patches'    
#folder_paths=[r'D:\MPI_CBG\data_plice\dataset\train\data',r'D:\MPI_CBG\data_plice\dataset\test\data',r'D:\MPI_CBG\data_plice\dataset\valid\data']
#lvl=2

dataset='cam'
patch_path=r'D:\MPI_CBG\camelyon16\patches'    
folder_paths=[r'D:\MPI_CBG\camelyon16\dataset\train\data',r'D:\MPI_CBG\camelyon16\dataset\test\data',r'D:\MPI_CBG\camelyon16\dataset\valid\data']
folder_paths=[r'D:\MPI_CBG\camelyon16\dataset\test\data']
lvl=3




file_names=[]
for folder_path in folder_paths:
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if name.endswith(".tif"):
                file_names.append(root+'\\'+name)
                
                
                
                
#file_names=file_names[16+30+216:]     
                
#file_names=file_names[36:]  
i=0                
for file_name in file_names:
    i+=1
    
    ###remove
#    if 'tumor_' not in file_name:
#        continue
    if '015' not in file_name:
        continue
    
    ###remove
    
    print(str(i) + '/'+ str(len(file_names)) + '  ' + file_name)
    
    fg_name=file_name.replace('\\data\\','\\fg\\')
    
    tmp = file_name.split('\\')
    train_test_valid=tmp[-3]
    name=tmp[-1][:-4]
    
    save_path_folder=patch_path + '\\' + train_test_valid + '\\' + name
    
    try:
        os.makedirs(save_path_folder)
    except:
        print('folder existuje')
    
    fg=imread(fg_name)>0
    img_s=imread_gdal(file_name,lvl)
    
    
    rgb_m,rgb_std,lab_m,lab_std=get_m_std(img_s,fg)
    
    
    np.save(save_path_folder +'\\rgb_m.npy', rgb_m)
    np.save(save_path_folder +'\\rgb_std.npy', rgb_std)
    np.save(save_path_folder +'\\lab_m.npy', lab_m)
    np.save(save_path_folder +'\\lab_std.npy', lab_std)
    
    
    
    he_mac_ref_name='HE_mac_' + dataset +'_ref.npy'
    he_vah_ref_name='HE_vah_' + dataset +'_ref.npy'
    mac_max,mac_mean,mac_std,HE_mac_inv,HE_mac,vah_max,vah_mean,vah_std,HE_vah_inv,HE_vah,mac_max_s,mac_mean_s,mac_std_s,HE_mac_inv_s,HE_mac_s,vah_max_s,vah_mean_s,vah_std_s,HE_vah_inv_s,HE_vah_s=get_stain_macenko_vah(img_s,fg,he_mac_ref_name,he_vah_ref_name)
    

    np.save(save_path_folder +'\\mac_max_s.npy', mac_max_s)
    np.save(save_path_folder +'\\mac_mean_s.npy', mac_mean_s)
    np.save(save_path_folder +'\\mac_std_s.npy', mac_std_s)
    np.save(save_path_folder +'\\HE_mac_inv_s.npy', HE_mac_inv_s)
    np.save(save_path_folder +'\\HE_mac_s.npy', HE_mac_s)
    
    np.save(save_path_folder +'\\vah_max_s.npy', vah_max_s)
    np.save(save_path_folder +'\\vah_mean_s.npy', vah_mean_s)
    np.save(save_path_folder +'\\vah_std_s.npy', vah_std_s)
    np.save(save_path_folder +'\\HE_vah_inv_s.npy', HE_vah_inv_s)
    np.save(save_path_folder +'\\HE_vah_s.npy', HE_vah_s)
    
    
    np.save(save_path_folder +'\\mac_max.npy', mac_max)
    np.save(save_path_folder +'\\mac_mean.npy', mac_mean)
    np.save(save_path_folder +'\\mac_std.npy', mac_std)
    np.save(save_path_folder +'\\HE_mac_inv.npy', HE_mac_inv)
    np.save(save_path_folder +'\\HE_mac.npy', HE_mac)
    
    np.save(save_path_folder +'\\vah_max.npy', vah_max)
    np.save(save_path_folder +'\\vah_mean.npy', vah_mean)
    np.save(save_path_folder +'\\vah_std.npy', vah_std)
    np.save(save_path_folder +'\\HE_vah_inv.npy', HE_vah_inv)
    np.save(save_path_folder +'\\HE_vah.npy', HE_vah)
    
    
    he_mac_ref_name='HE_mac_' + dataset +'_nolog_ref.npy'
    he_vah_ref_name='HE_vah_' + dataset +'_nolog_ref.npy'
    mac_max,mac_mean,mac_std,HE_mac_inv,HE_mac,vah_max,vah_mean,vah_std,HE_vah_inv,HE_vah,mac_max_s,mac_mean_s,mac_std_s,HE_mac_inv_s,HE_mac_s,vah_max_s,vah_mean_s,vah_std_s,HE_vah_inv_s,HE_vah_s=get_stain_macenko_vah(img_s,fg,he_mac_ref_name,he_vah_ref_name,logarithm=False)
    

    np.save(save_path_folder +'\\mac_max_s_nolog.npy', mac_max_s)
    np.save(save_path_folder +'\\mac_mean_s_nolog.npy', mac_mean_s)
    np.save(save_path_folder +'\\mac_std_s_nolog.npy', mac_std_s)
    np.save(save_path_folder +'\\HE_mac_inv_s_nolog.npy', HE_mac_inv_s)
    np.save(save_path_folder +'\\HE_mac_s_nolog.npy', HE_mac_s) #chybel nolog
    
    np.save(save_path_folder +'\\vah_max_s_nolog.npy', vah_max_s)
    np.save(save_path_folder +'\\vah_mean_s_nolog.npy', vah_mean_s)
    np.save(save_path_folder +'\\vah_std_s_nolog.npy', vah_std_s)
    np.save(save_path_folder +'\\HE_vah_inv_s_nolog.npy', HE_vah_inv_s)
    np.save(save_path_folder +'\\HE_vah_s_nolog.npy', HE_vah_s)
    
    
    
    
    
    
    
    
    np.save(save_path_folder +'\\mac_max_nolog.npy', mac_max)
    np.save(save_path_folder +'\\mac_mean_nolog.npy', mac_mean)
    np.save(save_path_folder +'\\mac_std_nolog.npy', mac_std)
    np.save(save_path_folder +'\\HE_mac_inv_nolog.npy', HE_mac_inv)
    np.save(save_path_folder +'\\HE_mac_nolog.npy', HE_mac)
    
    np.save(save_path_folder +'\\vah_max_nolog.npy', vah_max)
    np.save(save_path_folder +'\\vah_mean_nolog.npy', vah_mean)
    np.save(save_path_folder +'\\vah_std_nolog.npy', vah_std)
    np.save(save_path_folder +'\\HE_vah_inv_nolog.npy', HE_vah_inv)
    np.save(save_path_folder +'\\HE_vah_nolog.npy', HE_vah)
    
    
    
    
    
