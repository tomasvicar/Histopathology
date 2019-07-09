from skimage.io import imread
import numpy as np
import cv2
import gdal
from utils.gdal_fcns import imread_gdal,getsize_gdal


class Patcher():
    def __init__(self,img_name,fg_name,lbl_name,save_name,lvl,fg_lvl,fg_lbl_lvl_get,step,border,write_lvl,lbl_exist):
        self.img_name=img_name
        self.fg_name=fg_name
        self.lbl_name=lbl_name
        self.save_name=save_name
        self.lvl=lvl###image lvl to read
        self.fg_lvl=fg_lvl###fg lvl stored
        self.step=step### patch size
        self.border=border### overlap
        self.fg_lbl_lvl_get=fg_lbl_lvl_get #### what lvl of fg to create
        self.write_lvl=write_lvl ### what lvl will be writen
        self.lbl_exist=lbl_exist ### contain lbl or not (normal data has no lbl)
        
        img_shape=getsize_gdal(img_name,lvl)        
        fg_shape=getsize_gdal(fg_name)
        if self.lbl_exist:
            lbl_shape=getsize_gdal(lbl_name,lvl)
        else:
            lbl_shape=img_shape

        
        
        
        shape_min  = [np.min((fg_shape[0]*2**(self.fg_lvl-self.lvl),lbl_shape[0],img_shape[0])),
                      np.min((fg_shape[1]*2**(self.fg_lvl-self.lvl),lbl_shape[1],img_shape[1]))]
        
        corners=[]
        cx=0
        while cx<shape_min[0]-step-1: 
            cy=0
            while cy<shape_min[1]-step-1:
                corners.append([cx,cy])
                cy=cy+step-border-border-1
            cx=cx+step-border-border-1
            
          
            
        corners=np.round(np.array(corners)).astype(np.int)
        corners_fg=np.round(np.array(corners)/2**(self.fg_lvl-self.lvl)).astype(np.int)
        step_fg=np.round(step/2**(self.fg_lvl-self.lvl)).astype(np.int)
        
        
        fg=imread_gdal(fg_name) ##if fg fits to RAM
        
        use=np.ones(np.shape(corners)[0])
        counter=-1
        for c in corners_fg:
            counter+=1
#            fg_patch = imread_gdal(fg_name,position=(c[0],c[1],step_fg,step_fg)) ##if fg does not fit to RAM
            fg_patch=fg[c[1]:c[1]+step_fg,c[0]:c[0]+step_fg]
            if np.sum(fg_patch)==0:
                use[counter]=0
          
            
        self.corners=corners[np.where(use)[0],:]
        self.corners_save=np.round((self.corners+border)/2**(self.write_lvl-self.lvl)).astype(np.int)
        self.corners_fg=corners_fg[np.where(use)[0],:]

        self.step_save=np.round((self.step-border-border)/2**(self.write_lvl-self.lvl)).astype(np.int)
        self.step_fg=step_fg


        self.corner_num=np.shape(self.corners)[0]
        
        
        save_size_x=int(np.ceil(img_shape[0]/2**(self.write_lvl-self.lvl)))+1
        save_size_y=int(np.ceil(img_shape[1]/2**(self.write_lvl-self.lvl)))+1
        
        self.gdalObj_out = gdal.GetDriverByName('GTiff').Create(save_name, save_size_x,save_size_y, 1, gdal.GDT_Float32,['COMPRESS=LZW','BIGTIFF=YES','TILED=YES'])

        
        self.read_num=-1        
            
    def get_num_of_patches(self):
        return self.corner_num
    
    
    def read(self):
        self.read_num+=1
        
        c=self.corners[self.read_num]
        img=imread_gdal(self.img_name,level=self.lvl,position=(c[0],c[1],self.step,self.step))
        
        if self.lbl_exist:
            c=self.corners[self.read_num]
            lbl=imread_gdal(self.lbl_name,level=self.lvl,position=(c[0],c[1],self.step,self.step))
        else:
            lbl=np.zeros_like(img[:,:,0])
         
        c=self.corners_fg[self.read_num]
        fg=imread_gdal(self.fg_name,position=(c[0],c[1],self.step_fg,self.step_fg))    
        s=2**(self.fg_lvl-self.lvl)
        fg=cv2.resize(fg,None,fx=s, fy=s, interpolation = cv2.INTER_NEAREST) 
        
        fg=fg[self.border:-self.border,self.border:-self.border]
        lbl=lbl[self.border:-self.border,self.border:-self.border]
        
        s=2**(self.lvl-self.fg_lbl_lvl_get)
        lbl=cv2.resize(lbl,None,fx=s, fy=s, interpolation = cv2.INTER_NEAREST)  
        fg=cv2.resize(fg,None,fx=s, fy=s, interpolation = cv2.INTER_NEAREST)  
        
        
        
        return img,fg,lbl
        
        
    def write(self,data):
        
        c=self.corners_save[self.read_num]
        self.gdalObj_out.GetRasterBand(1).WriteArray(data,int(c[0]), int(c[1]))
        
        if self.read_num==(self.corner_num-1):
            self.close()

    
    

    
    def close(self):
        self.gdalObj_out.BuildOverviews("NEAREST", [2,4,8,16,32,64])
        self.gdalObj_out.FlushCache()                     # write to disk
        self.gdalObj_out = None
        
        