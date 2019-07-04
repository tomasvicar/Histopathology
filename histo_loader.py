import numpy as np
from torch.utils import data
import torch
import os
from  skimage.io import imread
import augmenter

import matplotlib.pyplot as plt
import time


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
       

        self.folder_path=[]
        for k in range(len(path_to_data)):
            self.folder_path.append(os.path.join(path_to_data[k],split))
        
        
        
        
        self.file_names=[]
        for k in self.folder_path:
            for root, dirs, files in os.walk(k):
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
            
            

            
        augmenters=[]
#        
#        
#        
#        plt.figure()
#        plt.imshow(data_im[0])
#        plt.show()
        
        augmenters.append(augmenter.ToFloat())
        augmenters.append(augmenter.RangeToRange((0,255),(0,1)))
    
        if self.split=='train':
            
            augmenters.append(augmenter.Rot90Flip())
            
            
#            if np.random.rand()>0.1:
#                augmenters.append(augmenter.HSVColorAugmenter(dh=0.01,ds=0.1,dv=0,mh=0,ms=0,mv=0))
#                augmenters.append(augmenter.ClipByValues((0,1)))
#                augmenters.append(augmenter.BrithnessContrastAugmenter(add=0.1,multipy=0.1))
#                augmenters.append(augmenter.ClipByValues((0,1)))
#                augmenters.append(augmenter.RGBColorAugmenter(dr=0.1,dg=0.1,db=0.1,mr=0.1,mg=0.1,mb=0.1,gr=0.1,gg=0.1,gb=0.1))
#                augmenters.append(augmenter.ClipByValues((0,1)))
                
#                augmenters.append(augmenter.BlurSharpAugmenter((-0.1,0.1)))
#                augmenters.append(augmenter.ElasticDeformer(np.shape(data_im[0])[:-1],grid_r=(5,10),mag_r=5))
                
            augmenters.append(augmenter.RandomCrop(in_size=256,out_size=246))
            
            
        else:
            augmenters.append(augmenter.CenterCrop(in_size=256,out_size=246))
            
            
        augmenters.append(augmenter.RangeToRange((0,1),(-1,1)))
        augmenters.append(augmenter.TorchFormat())

        

        
        data_im,data_mask=augmenter.augment_all(augmenters,data_im,data_mask)




        data_im=torch.cat(data_im,dim=0)
        if len(data_mask)>0:
            data_mask=data_mask[0]


#        plt.figure()
#        plt.imshow(data_im[0])
#        plt.show()
#        a=ffdfsdf
#        
        return data_im,data_mask,lbl
        
        
        
        
        
        
            

            
                    
        
