import numpy as np
import gdal





def imread_gdal_mask(data_name,level):
    
    level=level-1

    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        
    rOverview = rOverview.ReadAsArray(0,0, rOverview.XSize, rOverview.YSize) 
    
    return rOverview


def imread_gdal(data_name,level):
    
    level=level-1
    
    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    gOverview = gdalObj.GetRasterBand (2)
    bOverview = gdalObj.GetRasterBand (3)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        gOverview = gOverview.GetOverview(level)
        bOverview = bOverview.GetOverview(level)
    rOverview = rOverview.ReadAsArray(0,0, rOverview.XSize, rOverview.YSize) 
    gOverview = gOverview.ReadAsArray(0,0, gOverview.XSize, gOverview.YSize)  
    bOverview = bOverview.ReadAsArray(0,0, bOverview.XSize, bOverview.YSize)  
        
    return np.dstack((rOverview ,gOverview ,bOverview))




def get_patches_data_gdal(data_name,pos,patch_size,level):
    
    level=level-1
    patch_size_half=patch_size/2
    

    gdalObj = gdal.Open(data_name)
    
#     pos=pos[(1,0),:]

    
    nBands = gdalObj.RasterCount
    rOverview = gdalObj.GetRasterBand (1)
    gOverview = gdalObj.GetRasterBand (2)
    bOverview = gdalObj.GetRasterBand (3)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        gOverview = gOverview.GetOverview(level)
        bOverview = bOverview.GetOverview(level)
        
     
    overview = np.zeros((patch_size, patch_size, nBands), dtype=np.uint8)
    overview[:,:,0] = rOverview.ReadAsArray(int(pos[0]-patch_size_half), int(pos[1]-patch_size_half), int(patch_size), int(patch_size)) 
    overview[:,:,1] = gOverview.ReadAsArray(int(pos[0]-patch_size_half), int(pos[1]-patch_size_half), int(patch_size), int(patch_size)) 
    overview[:,:,2] = bOverview.ReadAsArray(int(pos[0]-patch_size_half), int(pos[1]-patch_size_half), int(patch_size), int(patch_size)) 
        
    return overview    
        
        
        
        
a=imread_gdal_mask(r'D:\MPI_CBG\camelyon16\dataset\valid\mask\tumor_068_mask.tif',3)      

c=imread_gdal(r'D:\MPI_CBG\camelyon16\dataset\valid\data\tumor_068.tif',3)   
  
b=get_patches_data_gdal(r'D:\MPI_CBG\camelyon16\dataset\valid\data\tumor_068.tif',[100,100],100,1)        
        
        
        
        
        