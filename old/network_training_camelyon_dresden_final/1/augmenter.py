
import numpy as np

from numpy.random import rand

import cv2

from scipy.interpolate import RectBivariateSpline


from scipy.interpolate import RegularGridInterpolator


from scipy.ndimage.filters import gaussian_filter

from scipy.ndimage.filters import laplace



#def augment_all(augmenters_list,imgs,masks):
#    for aug in augmenters_list:
#        for im in imgs:





def my_rot_flip(img,r):
    if r[0]:
        img=np.fliplr(img)
    if r[1]:
        img=np.flipud(img)
        
    img=np.rot90(img,k=r[2])        
    return img



class MatrixDeformer():
    def __init__(self,cols,rows,out_cols,out_rows,sr=0.05,gr=0.05,tr=0.001,dr=0):
        
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
    def __init__(self,img_cols,img_rows,grid_r=(5,10),mag_r=3):

        
        N=np.random.randint(grid_r[0],grid_r[1])
        mag=mag_r*rand()

        
        dx=mag-rand(N,N)*mag*2
        dy=mag-rand(N,N)*mag*2
        
        
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

    def augment(self,img,maska=False):
        
        if maska:
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



    
class ColorAugmenter():
    def __init__(self,colorchanels=3,gama_r=0.1,mult_r=0.1,plus_r=0.1): 
        
        self.gama=1+gama_r-(2*gama_r*np.random.rand(colorchanels))
        
        self.mult=1+mult_r-(2*mult_r*np.random.rand(colorchanels))
        
        self.plus=plus_r-(2*plus_r*np.random.rand(colorchanels))
        
    def augment(self,img):    
        
        for k in range(np.shape(img)[2]):
            img[:,:,k]=(((img[:,:,k]**self.gama[k])*self.mult[k]+self.plus[k]))
        
        
        return img
     
    def is_mask(self):
        return 0
        
class NoiseAugmenter():
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