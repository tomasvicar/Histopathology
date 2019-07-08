import gdal
from skimage.io import imread
import numpy as np
import cv2

class Patcher():
    def __init__(self,img_name,fg_name,lbl_name,save_name,lvl,step=400,border=100,write_scale=2,down_scale=1,fg_scale_down=8):
        self.img_name=img_name
        self.fg_name=fg_name
        self.lbl_name=lbl_name
        self.save_name=save_name
        self.step=step
        self.border=border
        self.lvl=lvl
        self.write_scale=write_scale
        self.down_scale=down_scale
        self.fg_scale_down=fg_scale_down
        
        self.border=border
        
        
#        step_s=int(step/fg_scale_down)
#        border_s=int(border*2/fg_scale_down)
        
        step_s=int(step/fg_scale_down)-1
        border_s=int(border*2/fg_scale_down)+1

        
        
        self.gdalObj_in = gdal.Open(img_name)
        tmp=self.gdalObj_in.GetRasterBand(1)
        self.rOverview = self.gdalObj_in.GetRasterBand (1)
        self.gOverview = self.gdalObj_in.GetRasterBand (2)
        self.bOverview = self.gdalObj_in.GetRasterBand (3)
        
        lvl_tmp=self.lvl-1
        if lvl_tmp>=0:
            self.rOverview = self.rOverview.GetOverview(lvl_tmp)
            self.gOverview = self.gOverview.GetOverview(lvl_tmp)
            self.bOverview = self.bOverview.GetOverview(lvl_tmp)
        
        
        
        if 'tumor' in lbl_name:
            self.gdalObj_lbl = gdal.Open(lbl_name)
            tmp2=self.gdalObj_lbl.GetRasterBand(1)
            self.lblOverview = self.gdalObj_lbl.GetRasterBand(1)
            lvl_tmp=self.lvl-1
            if lvl_tmp>=0:
                self.lblOverview = self.lblOverview.GetOverview(lvl_tmp)
        else:
            tmp2=tmp
        
        
        
        
        fg=imread(fg_name)
        fg=fg.T
        
        self.img_width  = np.min((np.shape(fg)[0],np.floor(tmp.XSize/(fg_scale_down*2**self.lvl)),np.floor(tmp2.XSize/(fg_scale_down*2**self.lvl))))
        self.img_height = np.min((np.shape(fg)[1],np.floor(tmp.YSize/(fg_scale_down*2**self.lvl)),np.floor(tmp2.YSize/(fg_scale_down*2**self.lvl))))

        
        corners=[]
        cx=0
        while cx<self.img_width-step_s-fg_scale_down-1: 
            cy=0
            while cy<self.img_height-step_s-fg_scale_down-1:
                corners.append([cx,cy])
                cy=cy+step_s-border_s
            cx=cx+step_s-border_s
            
            
        corners=np.round(np.array(corners))
        pouzit=np.ones(np.shape(corners)[0])
        citac=-1
        for c in corners:
            citac+=1
            kus = fg[c[0]:c[0]+step_s,c[1]:c[1]+step_s]
  
            if np.sum(kus)==0:
                pouzit[citac]=0
                    
        corners=corners[np.where(pouzit)[0],:]
        
        self.corners=corners*fg_scale_down
        
        self.prectene=-1
        
        
        
        
        
        
        
        
        
    
        xs=tmp.XSize
        ys=tmp.YSize
#        xs=np.max((np.shape(fg)[0]*scale_down,tmp.XSize))
        xs=int(np.ceil(xs/((self.write_scale*self.down_scale)*2**self.lvl)))+1
#        ys=np.max((np.shape(fg)[1]*scale_down,tmp.YSize))
        ys=int(np.ceil(ys/((self.write_scale*self.down_scale)*2**self.lvl)))+1
        
        self.gdalObj_out = gdal.GetDriverByName('GTiff').Create(save_name, xs,ys, 1, gdal.GDT_Float32,['COMPRESS=LZW','BIGTIFF=YES','TILED=YES'])
        
        

        self.gdalObj_fg = gdal.Open(fg_name)
        self.fgOverview = self.gdalObj_fg.GetRasterBand(1)
            
            


            
#            
        tmpx=self.corners[:,0]
        tmpy=self.corners[:,1]
#        
#        tmpxx=tmpx<=(tmp.XSize-self.step)
#        tmpyy=tmpy<=(tmp.YSize-self.step)
#        
#        use=np.bitwise_and(tmpxx,tmpyy)
#        self.corners=np.stack((tmpx[use],tmpy[use]),axis=1)
        
        
        self.corners=np.stack((tmpx,tmpy),axis=1)
        
        self.pocet=np.shape(self.corners)[0]
        
    def pocet_patchu(self):
        return self.pocet
    
    
    def read(self):
        self.prectene+=1
        print(self.img_name)
        print(str(self.prectene)+'//'+str(self.pocet))
        pozicex=self.corners[self.prectene,0]
        pozicey=self.corners[self.prectene,1]
        n=self.step
        
        patch=np.zeros((n, n,3))
        patch[:,:,0] = self.rOverview.ReadAsArray(int(pozicex), int(pozicey), int(n), int(n)) 
        patch[:,:,1] = self.gOverview.ReadAsArray(int(pozicex), int(pozicey), int(n), int(n)) 
        patch[:,:,2] = self.bOverview.ReadAsArray(int(pozicex), int(pozicey), int(n), int(n)) 
        
        fg=self.fgOverview.ReadAsArray(int(pozicex/self.fg_scale_down), int(pozicey/self.fg_scale_down), int(n/self.fg_scale_down), int(n/self.fg_scale_down)) 
        fg=cv2.resize(fg,None,fx=self.fg_scale_down, fy=self.fg_scale_down, interpolation = cv2.INTER_NEAREST)>0
        if self.border>0:
            fg=fg[self.border:-self.border,self.border:-self.border]
        
        if 'tumor' in self.lbl_name:
            lbl=self.lblOverview.ReadAsArray(int(pozicex), int(pozicey), int(n), int(n))>0
            if self.border>0:
                lbl=lbl[self.border:-self.border,self.border:-self.border]
        else:
            lbl=np.zeros(np.shape(fg),dtype=np.bool)
        
        fg=fg[::self.down_scale,::self.down_scale]
        lbl=lbl[::self.down_scale,::self.down_scale]
        
        return patch,fg,lbl
    
    
    def write(self,data):

        pozicex=self.corners[self.prectene,0]+self.border
        pozicey=self.corners[self.prectene,1]+self.border
        
        data=data[::self.write_scale,::self.write_scale]
        pozicex=np.floor(pozicex/(self.write_scale*self.down_scale))
        pozicey=np.floor(pozicey/(self.write_scale*self.down_scale))
        
        
        self.gdalObj_out.GetRasterBand(1).WriteArray(data,int(pozicex), int(pozicey))
    
    def close(self):
        self.gdalObj_out.BuildOverviews("NEAREST", [2,4,8,16,32,64])
        self.gdalObj_out.FlushCache()                     # write to disk
        self.gdalObj_out = None