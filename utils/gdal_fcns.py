import gdal
import numpy as np


def imread_gdal(name,level=0,position=None):
    
    '''
    
    position is bounding box  [x,y,length,width]
    
    '''
     
    level=level-1

    gdalObj = gdal.Open(name)
    
    
    nBands = gdalObj.RasterCount
    Overview = gdalObj.GetRasterBand(1)
    if level>=0:
        Overview = Overview.GetOverview(level)
        
    if position==None:
        position = (0,0, Overview.XSize, Overview.YSize)
        
#    print(Overview.XSize, Overview.YSize)
        
    img = np.zeros((position[3], position[2], nBands), dtype=np.uint8)
    
    img[:,:,0] = Overview.ReadAsArray(position[0], position[1],position[2],position[3])
    
    for k in range(2,nBands+1):
        Overview = gdalObj.GetRasterBand(k)
        if level>=0:
            Overview = Overview.GetOverview(level)
        
        img[:,:,k-1] = Overview.ReadAsArray(position[0], position[1],position[2],position[3])
        
    
    return np.squeeze(img)


def imwrite_gdal(data,name):
    
    if len(np.shape(data))>2:
        s=np.shape(data)[2]
    else:
        s=1
    
    dst_ds = gdal.GetDriverByName('GTiff').Create(name, np.shape(data)[1], np.shape(data)[0], s, gdal.GDT_Byte,['COMPRESS=LZW','BIGTIFF=YES','TILED=YES'])
    
    if len(np.shape(data))>2:
        kk=np.shape(data)[2]
    else:
        kk=1
        
    for k in range(kk):
        if len(np.shape(data))>2:
            dst_ds.GetRasterBand(k+1).WriteArray(data[:,:,k])   # write r-band to the raster
        else:
            dst_ds.GetRasterBand(k+1).WriteArray(data)   # write r-band to the raster

    dst_ds.BuildOverviews("NEAREST", [2,4,8,16,32,64])
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None







#import matplotlib.pyplot as plt
#
#name='/media/ubmi/DATA2/vicar/cam_dataset/train/data/normal_001.tif'
#name='/media/ubmi/DATA2/vicar/cam_dataset/train/mask/tumor_001_mask.tif'
#
#a=imread_gdal(name,level=0,position=(500,700,50,50))
#
#plt.imshow(a)




#name='/media/ubmi/DATA2/vicar/cam_dataset/train/data/normal_001.tif'
#name='/media/ubmi/DATA2/vicar/cam_dataset/train/mask/tumor_001_mask.tif'
#
#a=imread_gdal(name,level=4)
#
#imwrite_gdal(a,'test.tif')
