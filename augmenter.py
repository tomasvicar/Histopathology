
import numpy as np

from numpy.random import rand

import cv2

from scipy.interpolate import RectBivariateSpline


from scipy.interpolate import RegularGridInterpolator


from scipy.ndimage.filters import gaussian_filter

from scipy.ndimage.filters import laplace

import torch




from skimage.color import rgb2hsv
from skimage.color import hsv2rgb


def augment_all(augmenters_list,imgs,masks):
    for aug in augmenters_list:
        
        
        imgs_new=[]
        for im in imgs:
            imgs_new.append(aug.augment(im))
        
        if aug.is_mask and len(masks)>0:
            masks_new=[] 
            for im in masks:
                masks_new.append(aug.augment(im,mask=True))
        else:
            masks_new=masks
                
        imgs,masks=imgs_new,masks_new
    return imgs,masks 



class RangeToRange():
    def __init__(self,inr,outr):
        self.inr=inr
        self.outr=outr
    
    def augment(self,img,mask=False):
        return ((img-self.inr[0])/(self.inr[1]-self.inr[0]))*(self.outr[1]-self.outr[0])+self.outr[0]


    def is_mask(self):
        return 0
    
    
    
class ClipByValues():
    def __init__(self,vals):
        self.vals=vals
    
    def augment(self,img,mask=False):
        img[img<self.vals[0]]=self.vals[0]
        img[img>self.vals[1]]=self.vals[1]
        return img


    def is_mask(self):
        return 0




class ToFloat():
    def __init__(self):
        pass
    
    def augment(self,img,mask=False):
        return np.float32(img)


    def is_mask(self):
        return 1 
    
    
class TorchFormat():
    def __init__(self):
        pass
    
    def augment(self,img,mask=False):
        if mask:
            img=np.expand_dims(img ,axis=2)
        
        img=np.transpose(img,(2, 0, 1)).copy()
        img=torch.from_numpy(img)

        return img


    def is_mask(self):
        return 1 
    


class Rot90Flip():
    def __init__(self):
#        self.r=[np.random.randint(2),np.random.randint(2),np.random.randint(4)]
        self.r=[torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(2,(1,1)).view(-1).numpy(),torch.randint(4,(1,1)).view(-1).numpy()]
        
    def augment(self,img,mask=False):
        r=self.r
        if r[0]:
            img=np.fliplr(img)
        if r[1]:
            img=np.flipud(img)
                    
        img=np.rot90(img,k=r[2])  

        return img


    def is_mask(self):
        return 1 
    

class RandomCrop():
    def __init__(self,in_size,out_size):
#        self.r=[np.random.randint(in_size-out_size),np.random.randint(in_size-out_size)]
        self.r=[torch.randint(in_size-out_size,(1,1)).view(-1).numpy(),torch.randint(in_size-out_size,(1,1)).view(-1).numpy()]
        self.out_size=out_size
        
    def augment(self,img,mask=False):  
        r=self.r
        return img[r[0]:r[0]+self.out_size,r[1]:r[1]+self.out_size]


    def is_mask(self):
        return 1    

class CenterCrop():
    def __init__(self,in_size,out_size):
        self.r=[int(in_size-out_size),int(in_size-out_size)]
        self.out_size=out_size
        
    def augment(self,img,mask=False):  
        r=self.r
        return img[r[0]:r[0]+self.out_size,r[1]:r[1]+self.out_size]


    def is_mask(self):
        return 1   

