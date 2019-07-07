import numpy as np
import os
from skimage.io import imread
from utils.gdal_fcns import imread_gdal
import cv2

def clear_border(maskin,n):
    maskout=maskin
    maskout[0:n,:]=False
    maskout[-n:,:]=False
    maskout[:,0:n]=False
    maskout[:,-n:]=False
    return maskout



fg_lvl=4#pro camelyon
use_lvl=3#pro camelyon

patch_size_max=256
lbl_to_img_scale=2
border_to_clear=(2+int(np.ceil(patch_size_max/2/lbl_to_img_scale)))
num_of_idx_in_one_file=100_000


#path_to_data='/media/ubmi/DATA2/vicar/cam_dataset/train/data'
path_to_data='/media/ubmi/DATA2/vicar/cam_dataset/valid/data'

dataset='cam'
#dataset='lungs'


file_names=[]
for root, dirs, files in os.walk(path_to_data):
    for name in files:
        if name.endswith(".tif"):
            file_names.append(os.path.join(root,name))
#file_names=file_names[109:]


for k,file_name in enumerate(file_names):
    print(k)
    
    data_name=file_name
    
    fg_name=file_name.replace('/data/','/fg/')
        
        
    mask_name=file_name.replace('/data/','/mask/')
    mask_name=mask_name[:-4]+'_mask.tif'
        
    
    if data_name.find("normal")>=0:
        pouzit_masku=0
        
    elif data_name.find("tumor")>=0:
        pouzit_masku=1
    
    
        
    fg=imread(fg_name)
    if pouzit_masku:
        lbl=imread_gdal(mask_name,use_lvl)

    else:
        lbl=np.zeros([int(np.shape(fg)[0]*2**(fg_lvl-use_lvl)),int(np.shape(fg)[1]*2**(fg_lvl-use_lvl))], dtype=np.bool)
    
    fg=cv2.resize(fg,None,fx=2**(fg_lvl-use_lvl), fy=2**(fg_lvl-use_lvl), interpolation = cv2.INTER_NEAREST)>0#####plice
    
    
    fg=fg[:np.min([np.shape(lbl)[0],np.shape(fg)[0]]),:np.min([np.shape(lbl)[1],np.shape(fg)[1]])]
        
    lbl=lbl[:np.min([np.shape(lbl)[0],np.shape(fg)[0]]),:np.min([np.shape(lbl)[1],np.shape(fg)[1]])]
    

    
    tisue=np.bitwise_and(fg>0,lbl==0)
    if dataset=='lungs':
        tisue[lbl==1]=0
    tisue=clear_border(tisue,border_to_clear)
    position_tisue=np.where(tisue>0)
    position_tisue_num=len(position_tisue[0])
    
    
    
    tumor=lbl>0
    tumor=clear_border(tumor,border_to_clear)
    position_tumor=np.where(tumor>0)
    position_tumor_num=len(position_tumor[0])
    
    info= dict(fg_lvl=fg_lvl,use_lvl=use_lvl,patch_size_max=patch_size_max,
                    lbl_to_img_scale=lbl_to_img_scale,
                    position_tumor_num=position_tumor_num,
                    position_tisue_num=position_tisue_num,num_of_idx_in_one_file=num_of_idx_in_one_file)

    save_folder=file_name.replace('/data/','/idx/')[:-4]
    
    try:
        os.makedirs(save_folder)
    except:
        pass
    
    np.save(save_folder + '/' + 'info.npy',info,allow_pickle=True)
    
#    np:load(FDSF).flat[0]
    
    num_of_files=int(np.ceil(position_tisue_num/num_of_idx_in_one_file))
    for kk in range(num_of_files):
        if kk==num_of_files-1:
            tmp=[position_tisue[0][kk:kk+num_of_idx_in_one_file].astype(np.int32),position_tisue[1][kk:kk+num_of_idx_in_one_file].astype(np.int32)]
        else:
            tmp=[position_tisue[0][kk:kk+num_of_idx_in_one_file].astype(np.int32),position_tisue[1][kk:kk+num_of_idx_in_one_file].astype(np.int32)]
#        np.save(save_folder + '/' + 'idxs_tisue_'+ str(kk).zfill(6) +'.npy',tmp,allow_pickle=True)
        np.savez_compressed(save_folder + '/' + 'idxs_tisue_'+ str(kk).zfill(6) +'.npz',tmp,allow_pickle=True)
        
    num_of_files=int(np.ceil(position_tumor_num/num_of_idx_in_one_file))
    for kk in range(num_of_files):
        if kk==num_of_files-1:
            tmp=[position_tumor[0][kk:kk+num_of_idx_in_one_file].astype(np.int32),position_tumor[1][kk:kk+num_of_idx_in_one_file].astype(np.int32)]
        else:
            tmp=[position_tumor[0][kk:kk+num_of_idx_in_one_file].astype(np.int32),position_tumor[1][kk:kk+num_of_idx_in_one_file].astype(np.int32)]
#        np.save(save_folder + '/' + 'idxs_tumor_'+ str(kk).zfill(6) +'.npy',tmp,allow_pickle=True)   
        np.savez_compressed(save_folder + '/' + 'idxs_tumor_'+ str(kk).zfill(6) +'.npz',tmp,allow_pickle=True)  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        