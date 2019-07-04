import numpy as np
from torch.utils import data
import torch
import os
from  skimage.io import imread
import augmenter


class HistoDataset(data.Dataset):
    def __init__(self, split,path_to_data,s1=0,s2=0,s3=0,s1m=0,s2m=0,s3m=0,cn='no'):
        self.split=split
        self.s1=s1
        self.s2=s2
        self.s3=s3
        self.s1m=s1m
        self.s2m=s2m
        self.s3m=s3m
        self.cn=cn
       

        self.folder_path=os.path.join(path_to_data,split)
        
        self.subfolders = [f.path for f in os.scandir(self.folder_path) if f.is_dir() ] 
        
        
        self.file_names=[]
        for root, dirs, files in os.walk(self.folder_path):
            for name in files:
                if name.endswith(".tif") and name.startswith("data_1_"):
                    self.file_names.append(os.path.join(root,name))
                    
        self.num_of_img=len(self.file_names)

    def __len__(self):
        return self.num_of_img



    def __getitem__(self, index):
        
        name=self.file_names[index]
        
        name_s1=name
        name_s2=name.replace('data_1_','data_2_')
        name_s3=name.replace('data_2_','data_3_')
        
        name_s1m=name.replace('data_1_','mask_1_')
        name_s2m=name.replace('data_2_','mask_2_')
        name_s3m=name.replace('data_3_','mask_3_')
        
        lbl=int(name[name.rfind('data_1_')+1+6:name.rfind('data_1_')+2+6])
        
        
        data_im=[]
        data_mask=[]
        if self.s1:
            data_im.append(imread(name_s1))
        if self.s2:
            data_im.append(imread(name_s2))
        if self.s3:
            data_im.append(imread(name_s3))
        if self.s1m:
            data_mask.append(imread(name_s1m)>0)
        if self.s2m:
            data_mask.append(imread(name_s2m)>0)
        if self.s3m:
            data_mask.append(imread(name_s3m)>0)
            
            

        if self.cn=='no':    
            data_tmp=[]
            for img in data_im:
                data_tmp.append(img/(255/2)-1)
            data_im=data_tmp                

        
            
        rand_rot_flip=[np.random.randint(2),np.random.randint(2),np.random.randint(4)]   
        
        data_tmp=[]
        for im in data_im:
            im=np.float32(im)
            if  (self.split == 'train'):
                im=augmenter.my_rot_flip(im,rand_rot_flip)
            im=np.transpose(im,(2, 0, 1)).copy()
            im=torch.from_numpy(im)
            
            data_tmp.append(im)
        data_im=data_tmp
        
        data_tmp=[]
        for im in data_mask:
            im=np.float32(im)
            if  (self.split == 'train'):
                im=augmenter.my_rot_flip(im,rand_rot_flip)
            im=np.expand_dims(im ,axis=2)
            im=np.transpose(im,(2, 0, 1)).copy()
            im=torch.from_numpy(im)
            
            data_tmp.append(im)
        data_mask=data_tmp
        
        return data_im,data_mask,lbl
        
        
        
        
        
        
            

            
                    
        
