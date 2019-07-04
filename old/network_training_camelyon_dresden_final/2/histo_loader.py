
import numpy as np
from torch.utils import data
import torch
import os
#from scipy import misc
import skimage.io as misc

from skimage.color import rgb2lab
from skimage.color import lab2rgb
import matplotlib.pyplot as plt

from numpy.random import rand

import cv2

from scipy.interpolate import Rbf

from scipy.interpolate import griddata

from scipy.interpolate import interp2d

from scipy.interpolate import RectBivariateSpline


from scipy.interpolate import RegularGridInterpolator


from scipy.ndimage.filters import gaussian_filter

from scipy.ndimage.filters import laplace

import time




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
        
    def deform(self,img,mask=False):
        
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

    def deform(self,img,maska=False):
        
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





    
class ColorAugmenter():
    def __init__(self,colorchanels=3,gama_r=0.1,mult_r=0.1,plus_r=0.1): 
        
        self.gama=1+gama_r-(2*gama_r*np.random.rand(colorchanels))
        
        self.mult=1+mult_r-(2*mult_r*np.random.rand(colorchanels))
        
        self.plus=plus_r-(2*plus_r*np.random.rand(colorchanels))
        
    def augment(self,img):    
        
        for k in range(np.shape(img)[2]):
            img[:,:,k]=(((img[:,:,k]**self.gama[k])*self.mult[k]+self.plus[k]))
        
        
        return img
        
        
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







class HistoDataset(data.Dataset):
    def __init__(self, split,s=0,l=0,U=0,sm=0,lm=0,Um=0,cn=None,dataset=None):
        self.split=split
        self.s=s
        self.l=l
        self.U=U
        self.sm=sm
        self.lm=lm
        self.Um=Um
        self.cn=cn
       
        if dataset=='plice':
            if  (self.split == 'train'):
                self.folder_path=r'C:\Users\Tom\Desktop\deep_drazdany\histolky\both_datasets\dataset\patch_plice\train'
    
            elif (self.split == 'valid'):
                self.folder_path=r'C:\Users\Tom\Desktop\deep_drazdany\histolky\both_datasets\dataset\patch_plice\valid'
                
            elif (self.split == 'test'):
                self.folder_path=r'C:\Users\Tom\Desktop\deep_drazdany\histolky\both_datasets\dataset\patch_plice\test'
                
        elif dataset=='cam':
            if  (self.split == 'train'):
                self.folder_path=r'C:\data\vicar\patch_cam\train'
    
            elif (self.split == 'valid'):
                self.folder_path=r'C:\data\vicar\patch_cam\valid'
                
            elif (self.split == 'test'):
                self.folder_path=r'C:\data\vicar\patch_cam\test'
        
        self.subfolders = [f.path for f in os.scandir(self.folder_path) if f.is_dir() ] 
        
        
        self.file_names=[]
        for root, dirs, files in os.walk(self.folder_path):
            for name in files:
                if name.endswith(".tif") and name.startswith("data_0_"):
                    self.file_names.append(root+'\\'+name)
                    
        self.num_of_img=len(self.file_names)

    def __len__(self):
        return self.num_of_img



    def __getitem__(self, index):
        
        name=self.file_names[index]
        
        name_s=name
        name_l=name.replace('data_0_','data_1_')
        name_U=name.replace('data_0_','data_2_')
        
        name_sm=name.replace('data_0_','mask_0_')
        name_lm=name.replace('data_0_','mask_1_')
        name_Um=name.replace('data_0_','mask_2_')
        
        folder=name[:name.rfind('\\')+1] +'\\'
        
        lbl=int(name[name.rfind('data_0_')+1+6:name.rfind('data_0_')+2+6])
        
        
        data_im=[]
        data_mask=[]
        if self.s:
            data_im.append(misc.imread(name_s))
        if self.l:
            data_im.append(misc.imread(name_l))
        if self.U:
            data_im.append(misc.imread(name_U))
        if self.sm:
            data_mask.append(misc.imread(name_sm)>0)
        if self.lm:
            data_mask.append(misc.imread(name_lm)>0)
        if self.Um:
            data_mask.append(misc.imread(name_Um)>0)
            
            
        if self.cn=='R' :
            lab_m=np.load(folder +'lab_m.npy')
            lab_std=np.load(folder+'lab_std.npy')
            
            data_tmp=[]
            for img in data_im:
                lab=rgb2lab(img)
                for k in range(3):
                    lab[:,:,k]=(lab[:,:,k]-lab_m[k])/lab_std[k]
#                img=lab2rgb(lab)
                data_tmp.append(lab)
            data_im=data_tmp
            
        if self.cn=='no':    
            data_tmp=[]
            for img in data_im:
                data_tmp.append(img/255)
            data_im=data_tmp                
            
        if self.cn=='M':
            HE=np.load(folder+'HE_mac_s.npy')
            HE_inv=np.load(folder+'HE_mac_inv_s.npy')
            max=np.load(folder+'mac_max_s.npy')
            mean=np.load(folder+'mac_mean_s.npy')
            std=np.load(folder+'mac_stdc.npy')
            
            data_tmp=[]
            for img in data_im:
                data_v=np.reshape(img,(-1,3))
                data_v=-np.log10((data_v+1)/255)
                data_v=np.matmul(data_v,HE_inv)
                data_v[:,0]=data_v[:,0]/max[0]
                data_v[:,1]=data_v[:,1]/max[1]
                data_v=np.matmul(data_v,HE)
                
                data_v=np.exp(-data_v)
                
                img=np.reshape(data_v,np.shape(img))  
#        
                
                data_tmp.append(img)
            data_im=data_tmp  
         
            
        if self.cn=='V':
            HE=np.load(folder+'HE_vah.npy_s.npy')
            HE_inv=np.load(folder+'HE_vah_inv_s.npy')
            max=np.load(folder+'vah_max_s.npy')
            mean=np.load(folder+'vah_mean_s.npy')
            std=np.load(folder+'vah_stdc.npy')
            
            data_tmp=[]
            for img in data_im:
                data_v=np.reshape(img,(-1,3))
                data_v=-np.log10((data_v+1)/255)
                data_v=np.matmul(data_v,HE_inv)
                data_v[:,0]=data_v[:,0]/max[0]
                data_v[:,1]=data_v[:,1]/max[1]
                data_v=np.matmul(data_v,HE)
                
                data_v=np.exp(-data_v)
                
                img=np.reshape(data_v,np.shape(img))  
#        
                
                data_tmp.append(img)
            data_im=data_tmp  
            
            
            
            
         
        
#        plt.figure()
#        plt.imshow(img)
        
        
        
#        plt.figure()
#        plt.imshow(img)
        
        
        
        
#        plt.figure()
#        plt.imshow(img)
        
        
        
        
#        plt.figure()
#        plt.imshow(img)
        


        
#       
        
##        
#        plt.figure()
#        plt.imshow(img)
#   start_time = time.time()
#main()
#print("--- %s seconds ---" % (time.time() - start_time)
            
            
            
            
#        ra=rand()
        
        
#        if  (self.split == 'train') and ra>0.1:
#            img=data_im[0]
#            
#            
#            
#            start_time = time.time()
#            md=MatrixDeformer(320,320,96,96)
#            img=md.deform(img)
#    #        print("matrix--- %s seconds ---" % (time.time() - start_time))
#            
#
#            start_time = time.time()
#            ed=ElasticDeformer(96,96)
#            img=ed.deform(img)
#    #        print("elastic--- %s seconds ---" % (time.time() - start_time))            
#
#
#            start_time = time.time()
#            ca=ColorAugmenter()
#            img=ca.augment(img)
#    #        print("color--- %s seconds ---" % (time.time() - start_time))
#            
#            start_time = time.time()
#            ba=BlurSharpAugmenter()
#            img=ba.augment(img)
#    #        print("blur--- %s seconds ---" % (time.time() - start_time))
#            
#            start_time = time.time()
#            na=NoiseAugmenter(np.shape(img)[0],np.shape(img)[1])
#            img=na.augment(img)
#            data_im[0]=img
#    #        print("noise--- %s seconds ---" % (time.time() - start_time))
#        else:
#            
#            cols,rows,out_cols,out_rows=320,320,96,96
#            
#            ch=int(cols/2)
#            rh=int(rows/2)
#            coh=int(out_cols/2)
#            roh=int(out_rows/2)
#        
#            img=data_im[0]
#            img=img[ch-coh:ch+coh,rh-roh:rh+roh,:]
#        
#            data_im[0]=img



#        plt.figure()
#        plt.imshow(img)
            
            
        
        
        
            
        rand_rot_flip=[np.random.randint(2),np.random.randint(2),np.random.randint(4)]   
        
        data_tmp=[]
        for im in data_im:
            im=np.float32(im)
            if  (self.split == 'train'):
                im=my_rot_flip(im,rand_rot_flip)
            im=np.transpose(im,(2, 0, 1)).copy()
            im=torch.from_numpy(im)
            
            data_tmp.append(im)
        data_im=data_tmp
        
        data_tmp=[]
        for im in data_mask:
            im=np.float32(im)
            if  (self.split == 'train'):
                im=my_rot_flip(im,rand_rot_flip)
            im=np.expand_dims(im ,axis=2)
            im=np.transpose(im,(2, 0, 1)).copy()
            im=torch.from_numpy(im)
            
            data_tmp.append(im)
        data_mask=data_tmp
        
        return data_im,data_mask,lbl
        
        
        
        
        
        
            

            
                    
        
