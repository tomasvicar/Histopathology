import numpy as np
from torch.utils import data
import torch
import os
from utils.gdal_fcns import *



class HistoDataset(data.Dataset):
    def __init__(self, split='train',path_to_data='',level=0,get_mask=1,patch_size=256):
        self.split=split
        self.path_to_data=path_to_data
        self.level=level
        self.get_mask=get_mask
        self.patch_size=patch_size

       

        self.folder_path=os.path.join(path_to_data,split)
        
        
        
        
        self.file_names_normal=[]
        self.file_names_tumor=[]
        for k in self.folder_path:
            for root, dirs, files in os.walk(k):
                for name in files:
                    if name.endswith(".tif") and name.startswith("normal"):
                        self.file_names.append(os.path.join(root,name))
                    if name.endswith(".tif") and name.startswith("tumor"):
                        self.file_names.append(os.path.join(root,name))
         
        self.num_of_normal_imgs=len(self.file_names_normal)
        self.num_of_tumor_imgs=len(self.file_names_tumor)
        
            
        if self.split=='train':
            self.num_of_imgs=10000
        else:
            self.num_of_imgs=5000

    def __len__(self):
        return self.num_of_imgs



    def __getitem__(self, index):
        
        if not self.split=='train':
            state=np.random.get_state()
            np.random.seed(index)
            
        
        lbl=np.random.randint(0,2)
        
        if lbl:
            img_index=np.random.randint(self.num_of_tumor_imgs)
            img_name=self.file_names_tumor[img_index]
        else:
            img_index=np.random.randint(self.num_of_nromal_imgs)
            img_name=self.file_names_normal[img_index]
            
        center_position=self.get_pixel_position(img_name,lbl) 
        position=(int(center_position[0]-patch_size/2), int(center_position[1]-patch_size/2), int(patch_size), int(patch_size))
        
        img=imread_gdal(name,level=self.level,position=position)
        if get_mask:
            mask=imread_gdal(name,level=self.level,position=position)
            
        
        if not self.split=='train':
            np.random.set_state(state)
        
        return img,mask,lbl
        
        
     def get_pixel_position(self,img_name,lbl):
         pass
        
        
        
            

            
                    
        
