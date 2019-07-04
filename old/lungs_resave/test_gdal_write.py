import javabridge
import bioformats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from read_labeling2 import read_lbl2
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.color import rgb2lab
import gc
import gdal
import time
from skimage.io import imsave

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
        
    return np.stack((rOverview ,gOverview ,bOverview),axis=2)



def write_gdal(data,name):
    
    dst_ds = gdal.GetDriverByName('GTiff').Create(name, np.shape(data)[1], np.shape(data)[0], 3, gdal.GDT_Byte)
    
    for k in np.shape(data)[2]:
        dst_ds.GetRasterBand(k+1).WriteArray(data[:,:,k])   # write r-band to the raster

    dst_ds.BuildOverviews("NEAREST", [2,4,8,16,32,64])
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None







javabridge.start_vm(class_path=bioformats.JARS)



patch_size=96
patch_size_l=320
kolik=500
#comp='lzma'
comp=6

folder=r'D:\MPI_CBG\data_plice\data\tumor'



label_dict=	{'??': 1, 'solid':2,'diffuse':3}





files=os.listdir(folder)
files_labeling = [i for i in files if i.endswith('.labeling')]

image_name=files_labeling[0]

print(image_name)

try:
    os.makedirs(folder_save+'\\'+image_name[0:-13])
except:
    print('folder existuje')

lbl_path = folder + '\\' + image_name

img_path=folder + '\\' + image_name[0:16] + '.czi'



lbl = read_lbl2(lbl_path,label_dict)


lbl_size=lbl.shape


image_info = bioformats.get_omexml_metadata(img_path)
omxml_data=bioformats.OMEXML(image_info)

channel_count = omxml_data.image_count
ii=0
for i in range(channel_count):
    pixels = omxml_data.image(i).Pixels
    y = pixels.SizeX
    x = pixels.SizeY
#        print(x,y)
    if (lbl_size[0]==x) and (lbl_size[1]==y):
        ii=i
        break
if ii==0:
    raise NameError('chanel not exist')



img_l = bioformats.load_image(img_path,z=0,t=0,series=ii-1, rescale=False)


#gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
#LZW
#    ['COMPRESS=LZW']

start = time.time()

dst_ds = gdal.GetDriverByName('GTiff').Create('myGeoTIFF.tif', np.shape(img_l)[1], np.shape(img_l)[0], 3, gdal.GDT_Byte,['COMPRESS=LZW'])

dst_ds.GetRasterBand(1).WriteArray(img_l[:,:,0])   # write r-band to the raster
dst_ds.GetRasterBand(2).WriteArray(img_l[:,:,1])   # write g-band to the raster
dst_ds.GetRasterBand(3).WriteArray(img_l[:,:,2])   # write b-band to the raster
dst_ds.BuildOverviews("NEAREST", [2,4,8,16,32,64])
dst_ds.FlushCache()                     # write to disk
dst_ds = None

end = time.time()
print(end - start)


start = time.time()

#img_s = bioformats.load_image(img_path,z=0,t=0,series=ii, rescale=False)

img_s_2=imread_gdal('myGeoTIFF.tif',0)

end = time.time()
print(end - start)

#  
#
#print(gdalObj.GetMetadata())


#gdalObj = gdal.Open('myGeoTIFF.tif')
#md = gdalObj.GetMetadata('IMAGE_STRUCTURE')
