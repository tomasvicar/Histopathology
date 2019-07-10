
import numpy as np


import cv2

from scipy.interpolate import RectBivariateSpline


from scipy.interpolate import RegularGridInterpolator


from scipy.ndimage.filters import gaussian_filter

from scipy.ndimage.filters import laplace

import torch


from skimage.color import rgb2hsv
from skimage.color import hsv2rgb



def rand():
    return torch.rand(1).numpy()[0]



def augment_all(augmenters_list,imgs,masks):
    for aug in augmenters_list:
        
        
        imgs_new=[]
        for im in imgs:
            imgs_new.append(aug.augment(im))
        
        if aug.is_mask() and len(masks)>0:
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



class BlurSharpAugmenter():
    def __init__(self,bs_r=(-0.5,0.5)): 
        r=1-2*rand()
        
    
        
        if r<=0:
            self.type='s'
            self.par=bs_r[0]*r
            
        if r>0:
            self.type='b'
            self.par=bs_r[1]*r
        
        
        
    def augment(self,img):   
        if self.type=='b':
            for k in range(np.shape(img)[2]):
                img[:,:,k]=gaussian_filter(img[:,:,k],self.par)
        
        if self.type=='s':
            for k in range(np.shape(img)[2]):
                img[:,:,k]=img[:,:,k]-self.par*laplace(img[:,:,k])
        
    
        return img

    def is_mask(self):
        return 0 
    
    
    
    
    
class HSVColorAugmenter():
    def __init__(self,dh=0,ds=0,dv=0,mh=0,ms=0,mv=0): 
        self.dh=(1-2*rand())*dh
        self.ds=(1-2*rand())*ds
        self.dv=(1-2*rand())*dv
        
        self.mh=1-(1-2*rand())*mh
        self.ms=1-(1-2*rand())*ms
        self.mv=1-(1-2*rand())*mv
        


    def augment(self,img):
        dt=img.dtype
        
#        img=rgb2hsv(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        h=img[:,:,0]
        s=img[:,:,1]
        v=img[:,:,2]
        
        h=(h)*self.mh+self.dh
        s=(s)*self.ms+self.ds
        v=(v)*self.mv+self.dv
        
        
        img=np.stack([h,s,v],axis=2)
        
#        img=hsv2rgb(img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    
        return img.astype(dt)

    def is_mask(self):
        return 0 
    
  
    
    
class RGBColorAugmenter():
    def __init__(self,dr=0,dg=0,db=0,mr=0,mg=0,mb=0,gr=0,gg=0,gb=0): 
        self.dr=(1-2*rand())*dr
        self.dg=(1-2*rand())*dg
        self.db=(1-2*rand())*db
        
        self.mr=1-(1-2*rand())*mr
        self.mg=1-(1-2*rand())*mg
        self.mb=1-(1-2*rand())*mb
        
        self.gr=1-(1-2*rand())*gr
        self.gg=1-(1-2*rand())*gg
        self.gb=1-(1-2*rand())*gb


    def augment(self,img):   
        
        r=img[:,:,0]
        g=img[:,:,1]
        b=img[:,:,2]
        
        r=(r**self.gr)*self.mr+self.dr
        g=(g**self.gg)*self.mg+self.dg
        b=(b**self.gb)*self.mb+self.db
        
        img=np.stack([r,g,b],axis=2)
    
        return img

    def is_mask(self):
        return 0 
    
    
    
class BrithnessContrastAugmenter():
    def __init__(self,add=0.1,multipy=0.1): 
        self.add=(1-2*rand())*add
        self.multipy=1+(1-2*rand())*multipy


    def augment(self,img):
    
        return (img*self.multipy)+self.add

    def is_mask(self):
        return 0 

