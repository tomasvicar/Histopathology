
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
        self.r=[np.random.randint(2),np.random.randint(2),np.random.randint(4)]
        
        
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
        self.r=[np.random.randint(in_size-out_size),np.random.randint(in_size-out_size)]
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

class MatrixDeformer():
    def __init__(self,cols,rows,out_cols,out_rows,sr=0.05,gr=0.05,tr=0.001,dr=0):
        
        #sr = scales
        #gr = shears
        #tr = tilt
        #dr = translation
        
        sx=(1-sr)+sr*2*rand()
        sy=(1-sr)+sr*2*rand()
        
        gx=(0-gr)+gr*2*rand()
        gy=(0-gr)+gr*2*rand()
        
        tx=(0-tr)+tr*2*rand()
        ty=(0-tr)+tr*2*rand()
        
        dx=(0-dr)+dr*2*rand()
        dy=(0-dr)+dr*2*rand()
        
        t=2*np.pi*rand()
        
        
        M=np.array([[sx, gx, dx], [gy, sy, dy],[tx, ty, 1]])
        
        
        R=cv2.getRotationMatrix2D((cols / 2, rows / 2), t/np.pi*180, 1)
        R=np.concatenate((R,np.array([[0,0,1]])),axis=0)
        
        
        self.cols=cols
        self.rows=rows
        
        self.out_cols=out_cols
        self.out_rows=out_rows
        self.matrix= np.matmul(R,M)
        
    def augment(self,img,mask=False):
        
        if mask:
            flags=cv2.INTER_NEAREST
        else:
            flags=cv2.INTER_LINEAR
        
        
        img = cv2.warpPerspective(img,self.matrix, (self.cols,self.rows),flags=flags)
        
        
        ch=int(self.cols/2)
        rh=int(self.rows/2)
        coh=int(self.out_cols/2)
        roh=int(self.out_rows/2)
        
        img=img[ch-coh:ch+coh,rh-roh:rh+roh,:]
        
        return img
    
    
    def is_mask(self):
        return 1
    
    
    

class ElasticDeformer():
    def __init__(self,shape,grid_r=(5,10),mag_r=3):
        
        img_cols,img_rows=shape
        
        N=np.random.randint(grid_r[0],grid_r[1])
        mag=mag_r*rand()

        
        dx=mag-rand(N,N)*mag*2
        dy=mag-rand(N,N)*mag*2
        
        #dont move with borders
        dx[0,:]=0
        dx[-1,:]=0
        dx[:,0]=0
        dx[:,-1]=0
        
        dy[0,:]=0
        dy[-1,:]=0
        dy[:,0]=0
        dy[:,-1]=0
        
        
        x,y=np.linspace(0,img_cols,N),np.linspace(0,img_rows,N)
        
        fx = RectBivariateSpline(x, y, dx)
        fy = RectBivariateSpline(x, y, dy)
        
        
        
        [x,y]=np.meshgrid(np.arange(0,img_cols),np.arange(0,img_rows))
        x=x.flatten()
        y=y.flatten()
        
        self.x=np.arange(0,img_cols)
        self.y=np.arange(0,img_rows)
        
        
        self.x_def=x+fx(self.x,self.y).flatten()
        self.y_def=y+fy(self.x,self.y).flatten()
        
        
        
        
#        fx = Rbf(x, y, dx,function=function)
#        fy = Rbf(x, y, dy,function=function)
#        
#        
#        self.x_def=self.x+fx(self.x,self.y)
#        self.y_def=self.y+fy(self.x,self.y)
        
        
        self.x_def[self.x_def<0]=0
        self.x_def[self.x_def>img_rows-1]=img_rows-1
        self.y_def[self.y_def<0]=0
        self.y_def[self.y_def>img_rows-1]=img_rows-1
        
        
        self.x_def,self.y_def=self.y_def,self.x_def

    def augment(self,img,mask=False):
        
        if mask:
            method='nearest'
        else:
            method='linear'
    
        
        r=len(self.x)
        c=len(self.y)
        
        f=RegularGridInterpolator((self.x,self.y),img[:,:,0],method=method)
        img[:,:,0]=np.reshape(f(np.stack((self.x_def,self.y_def),axis=1)),(r,c))
        
        f=RegularGridInterpolator((self.x,self.y),img[:,:,1],method=method)
        img[:,:,1]=np.reshape(f(np.stack((self.x_def,self.y_def),axis=1)),(r,c))
        
        f=RegularGridInterpolator((self.x,self.y),img[:,:,2],method=method)
        img[:,:,2]=np.reshape(f(np.stack((self.x_def,self.y_def),axis=1)),(r,c))
        

        return img

    def is_mask(self):
        return 1



    

        
class NoiseAugmenterGaussIllumination():
    def __init__(self,img_cols,img_rows,colorchanels=3,il_r=0.1,gau_r=0.05): 
        
        [X,Y]=np.meshgrid(np.linspace(0,1,img_cols),np.linspace(0,1,img_rows));
        il_r_1=il_r;
        il_r_2=il_r**2;
        
        a=il_r_1-2*il_r_1*rand();
        b=il_r_1-2*il_r_1*rand();
        
        c=il_r_2-2*il_r_2*rand();
        d=il_r_2-2*il_r_2*rand();
        e=il_r_2-2*il_r_2*rand();
        
        il=1+a*X+b*Y+c*X*Y+d*X**2++e*Y**2
        self.il=np.repeat(np.expand_dims(il,axis=2),colorchanels,axis=2)
        
        self.noise=gau_r*rand()*np.random.randn(img_cols,img_rows,colorchanels)
        
        
    def augment(self,img):    
    
        return img*self.il+self.noise        
    

    def is_mask(self):
        return 0 
    
    
    
    


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