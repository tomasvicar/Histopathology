import numpy as np
import gdal
import os
import cv2


def imread_gdal_mask(data_name,level,aaa):
    
    level=level-1

    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        
#    rOverview = rOverview.ReadAsArray(0,0, rOverview.XSize, rOverview.YSize) 
    rOverview = rOverview.ReadAsArray(aaa[0],aaa[1], aaa[2], aaa[3]) 
    
    return rOverview



data_path=r'D:\MPI_CBG\data_plice\dataset\train\mask'
lvl=0



file_names=[]
for root, dirs, files in os.walk(data_path):
    for name in files:
        if name.endswith(".tif"):
            file_names.append(root+'\\'+name)
            


f=np.array(0,dtype=np.float64)
b=np.array(0,dtype=np.float64)

#file_names=file_names[4:]

for img_filename in file_names:
    
    print(img_filename)
    
    fg_name=img_filename
    fg_name=fg_name.replace('\\mask\\','\\fg\\')
    fg_name=fg_name.replace('_mask.tif','.tif')

    
    reader_result=gdal.Open(img_filename).GetRasterBand(1)
    if lvl-1>=0:
        reader_result = reader_result.GetOverview(lvl-1)
    
    x=reader_result.XSize
    y=reader_result.YSize

    step=1000
    for k in np.arange(0,x,step):
        for kk in np.arange(0,y,step):
            n=step
            m=step
            
            if k+n>x:
                n=n-(x-k+n)-1
            
            if kk+m>y:
                m=m-(y-kk+m)-1 
                
                
            if m<0 or n<0:
                continue
                
            result=imread_gdal_mask(img_filename,lvl,[int(k),int(kk),int(m),int(n)])
            
            fg=imread_gdal_mask(fg_name,lvl,[int(np.floor(k/8)), int(np.floor(kk/8)), int(np.floor(m/8)), int(np.floor(n/8))]) 
            
            fg=cv2.resize(fg,None,fx=8, fy=8, interpolation = cv2.INTER_NEAREST)>0
           
            
                
            f+=np.sum(np.bitwise_and(result>0,fg))
            b+=np.sum(np.bitwise_and(result==0,fg))

            
print('f:   '+ str(f))
print('b:   '+ str(b))
           

print('f/(b+f):'   + str(f/(b+f)))


