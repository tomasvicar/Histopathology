import gdal
from skimage.io import imread
import numpy as np
import cv2


#def write_gdal(data,name):
#    
#    dst_ds = gdal.GetDriverByName('GTiff').Create(name, np.shape(data)[1], np.shape(data)[0], 3, gdal.GDT_Byte,['COMPRESS=LZW','BIGTIFF=YES','TILED=YES'])
#    
#    if len(np.shape(data))>2:
#        kk=np.shape(data)[2]
#    else:
#        kk=1
#    for k in range(kk):
#        if len(np.shape(data))>2:
#            dst_ds.GetRasterBand(k+1).WriteArray(data[:,:,k])   # write r-band to the raster
#        else:
#            dst_ds.GetRasterBand(k+1).WriteArray(data)   # write r-band to the raster
#
#    dst_ds.BuildOverviews("NEAREST", [2,4,8,16,32,64])
#    dst_ds.FlushCache()                     # write to disk
#    dst_ds = None




class Patcher():
    def __init__(self,img_name,fg_name,lbl_name,save_name,lvl,step=400,border=100,write_scale=2):
        self.img_name=img_name
        self.fg_name=fg_name
        self.lbl_name=lbl_name
        self.save_name=save_name
        self.step=step
        self.border=border
        self.lvl=lvl
        self.write_scale=write_scale
        
        
        self.border=border
        
        scale_down=8
        
        step_s=int(step/scale_down)
        border_s=int(border*2/scale_down)
        
        if np.round(step_s)!=step_s or np.round(border_s)!=border_s:
            raise Exception('is not divisible by scale_down')
        
        
        self.gdalObj_in = gdal.Open(img_name)
        tmp=self.gdalObj_in.GetRasterBand(1)
        self.rOverview = self.gdalObj_in.GetRasterBand (1)
        self.gOverview = self.gdalObj_in.GetRasterBand (2)
        self.bOverview = self.gdalObj_in.GetRasterBand (3)
        
        lvl_tmp=self.lvl-1
        if lvl_tmp>=0:
            self.rOverview = self.rOverview.GetOverview(self.lvl)
            self.gOverview = self.gOverview.GetOverview(self.lvl)
            self.bOverview = self.bOverview.GetOverview(self.lvl)
        
        
        
        
        
        
        fg=imread(fg_name)
        fg=fg.T
        
        self.img_width  = np.min((np.shape(fg)[0],np.floor(tmp.XSize/(scale_down*2**self.lvl))))
        self.img_height = np.min((np.shape(fg)[1],np.floor(tmp.YSize/(scale_down*2**self.lvl))))

        
        corners=[]
        cx=0
        while cx<self.img_width-step_s-scale_down-1: 
            cy=0
            while cy<self.img_height-step_s-scale_down-1:
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
        
        self.corners=corners*scale_down
        
        self.prectene=-1
        
        
        
        
        
        
        
        
        
    
        xs=tmp.XSize
        ys=tmp.YSize
#        xs=np.max((np.shape(fg)[0]*scale_down,tmp.XSize))
        xs=int(np.ceil(xs/(self.write_scale*2**self.lvl)))
#        ys=np.max((np.shape(fg)[1]*scale_down,tmp.YSize))
        ys=int(np.ceil(ys/(self.write_scale*2**self.lvl)))
        
        self.gdalObj_out = gdal.GetDriverByName('GTiff').Create(save_name, xs,ys, 1, gdal.GDT_Float32,['COMPRESS=LZW','BIGTIFF=YES','TILED=YES'])
        
        
        
        
        self.gdalObj_fg = gdal.Open(fg_name)
        self.fgOverview = self.gdalObj_fg.GetRasterBand(1)
#        if self.lvl>=0:
#            self.fgOverview = self.fgOverview.GetOverview(self.lvl)
            
            
        if 'tumor' in lbl_name:
            self.gdalObj_lbl = gdal.Open(lbl_name)
            self.lblOverview = self.gdalObj_lbl.GetRasterBand(1)
            lvl_tmp=self.lvl-1
            if lvl_tmp>=0:
                self.lblOverview = self.lblOverview.GetOverview(lvl_tmp)

            
            
        tmpx=self.corners[:,0]
        tmpy=self.corners[:,1]
        
        tmpxx=tmpx<=(tmp.XSize-self.step)
        tmpyy=tmpy<=(tmp.YSize-self.step)
        
        use=np.bitwise_and(tmpxx,tmpyy)
        self.corners=np.stack((tmpx[use],tmpy[use]),axis=1)
        
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
        
        fg=self.fgOverview.ReadAsArray(int(pozicex/8), int(pozicey/8), int(n/8), int(n/8)) 
        fg=cv2.resize(fg,None,fx=8, fy=8, interpolation = cv2.INTER_NEAREST)>0
        if self.border>0:
            fg=fg[self.border:-self.border,self.border:-self.border]
        
        if 'tumor' in self.lbl_name:
            lbl=self.lblOverview.ReadAsArray(int(pozicex), int(pozicey), int(n), int(n))>0
            if self.border>0:
                lbl=lbl[self.border:-self.border,self.border:-self.border]
        else:
            lbl=np.zeros(np.shape(fg),dtype=np.bool)
        
        
        
        return patch,fg,lbl
    
    
    def write(self,data):

        pozicex=self.corners[self.prectene,0]+self.border
        pozicey=self.corners[self.prectene,1]+self.border
        
        data=data[::self.write_scale,::self.write_scale]
        pozicex=np.floor(pozicex/self.write_scale)
        pozicey=np.floor(pozicey/self.write_scale)
        
#        print(self.gdalObj_out.GetRasterBand(1).XSize,self.gdalObj_out.GetRasterBand(1).YSize)
        
        self.gdalObj_out.GetRasterBand(1).WriteArray(data,int(pozicex), int(pozicey))
    
    def close(self):
        self.gdalObj_out.BuildOverviews("NEAREST", [2,4,8,16,32,64])
        self.gdalObj_out.FlushCache()                     # write to disk
        self.gdalObj_out = None
            
            
            
            