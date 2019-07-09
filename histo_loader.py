import numpy as np
from torch.utils import data
import torch
import os
from utils.gdal_fcns import imread_gdal

import augmenter



class HistoDataset(data.Dataset):
    def __init__(self, split='train',path_to_data='',level=0,get_mask=1,patch_size=256):
        self.split=split
        self.path_to_data=path_to_data

        self.level=level
        self.get_mask=get_mask
        self.patch_size=patch_size

        
        
        
        self.file_names_normal=[]
        self.file_names_tumor=[]
        for root, dirs, files in os.walk(self.path_to_data):
            for name in files:
                if name.endswith(".tif") and name.startswith("normal"):
                    self.file_names_normal.append(os.path.join(root,name))
                if name.endswith(".tif") and name.startswith("tumor"):
                    self.file_names_tumor.append(os.path.join(root,name))
         
        self.num_of_normal_imgs=len(self.file_names_normal)
        self.num_of_tumor_imgs=len(self.file_names_tumor)
        
            
        if self.split=='train':
            self.num_of_imgs=10000
        else:
            self.num_of_imgs=2000

    def __len__(self):
        return self.num_of_imgs



    def __getitem__(self, index):
#        
        if not self.split=='train':
            np.random.seed(index)
#            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            
        if not self.split=='train':
            lbl=np.random.randint(0,2)
        else:
            lbl=torch.randint(2,(1,1)).view(-1).numpy()
#        print(lbl)
        
        if lbl:
            if not self.split=='train':
                img_index=np.random.randint(self.num_of_tumor_imgs)
            else:
                img_index=int(torch.randint(self.num_of_tumor_imgs,(1,1)).view(-1).numpy())
            img_name=self.file_names_tumor[img_index]
        else:
            if not self.split=='train':
                img_index=np.random.randint(self.num_of_normal_imgs)
            else:
                img_index=int(torch.randint(self.num_of_normal_imgs,(1,1)).view(-1).numpy())
            img_name=self.file_names_normal[img_index]
            
        mask_name=img_name.replace(os.sep+'data'+os.sep,os.sep+'mask'+os.sep).replace('.tif','_mask.tif')
            
        center_position=self.get_pixel_position(img_name,lbl,self.level) 
        
        center_position=[center_position[1],center_position[0]]
        
#        print(center_position)
        
        position=(int(center_position[0]-self.patch_size/2), int(center_position[1]-self.patch_size/2),
                  int(self.patch_size), int(self.patch_size))
        
#        print(position)
        
        img=imread_gdal(img_name,level=self.level,position=position)
        if self.get_mask:
            if 'tumor_' in mask_name:
                mask=imread_gdal(mask_name,level=self.level,position=position)
            else:
                mask=np.zeros((self.patch_size,self.patch_size), dtype=np.uint8)

            
        augmenters=[]    
        augmenters.append(augmenter.ToFloat())
        augmenters.append(augmenter.RangeToRange((0,255),(-1,1)))
        if self.split=='train':
            augmenters.append(augmenter.Rot90Flip())
        augmenters.append(augmenter.TorchFormat())
        
        img,mask=augmenter.augment_all(augmenters,[img],[mask])
        img,mask=img[0],mask[0]
            
        

            
        lbl=torch.from_numpy(np.array(np.float32(lbl)))
        
        return img,mask,lbl
        
        
    def get_pixel_position(self,img_name,lbl,lvl):
        save_folder=img_name.replace(os.sep+'data'+os.sep,os.sep+'idx'+os.sep)[:-4]
   
        info=np.load(save_folder + os.sep + 'info.npy',allow_pickle=True).flat[0]
        num_of_idx_in_one_file=info.get('num_of_idx_in_one_file')
         
         
        if lbl:
            position_num=info.get('position_tumor_num')
        else:
            position_num=info.get('position_tisue_num')
        
        if not self.split=='train':
            idx_num=np.random.randint(0,position_num)
        else:
            idx_num=torch.randint(position_num,(1,1)).view(-1).numpy()
        
        file_num= int(np.floor(idx_num/num_of_idx_in_one_file) )
        idx_in_file=idx_num%num_of_idx_in_one_file
        
        if lbl:
            positions=np.load(save_folder + os.sep + 'idxs_tumor_'+ str(file_num).zfill(6) +'.npz',allow_pickle=True)
        else:
            positions=np.load(save_folder + os.sep + 'idxs_tisue_'+ str(file_num).zfill(6) +'.npz',allow_pickle=True)
            
        positions=positions.f.arr_0
        position=[positions[0][idx_in_file],positions[1][idx_in_file]]
        
        use_lvl=info.get('use_lvl')
        
#        print(2**(use_lvl-lvl))
        
        if not self.split=='train':
            rx=np.random.randint(2**(use_lvl-lvl))
            ry=np.random.randint(2**(use_lvl-lvl))
        else:
            rx=int(torch.randint(2**(use_lvl-lvl),(1,1)).view(-1).numpy())
            ry=int(torch.randint(2**(use_lvl-lvl),(1,1)).view(-1).numpy())
        
        position=[position[0]*2**(use_lvl-lvl)+rx,position[1]*2**(use_lvl-lvl)+ry]
            
        return position           
        
