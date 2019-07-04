import javabridge
import bioformats
import numpy as np
import os
import gdal
from read_labeling2 import read_lbl2
import gc
import cv2
import matplotlib.pyplot as plt


javabridge.start_vm(class_path=bioformats.JARS)

path=r'D:\MPI_CBG\data_plice\data\normal\2017_10_24__0202.czi'

out_path='test.tif'

#ImageReader = bioformats.formatreader.make_image_reader_class()
#reader = ImageReader()
#reader.setId(path)
#(reader.getSizeY(), reader.getSizeX())
#data = reader.openBytesXYWH(0,0,0,10000,10000)
#
#data=np.reshape(data,(10000,10000,3))
#
#plt.imshow(data)

#
#data=bioformats.load_image(path,z=0,t=0,series=29, rescale=False,XYWH= (0,0,0,10000,10000))
#plt.imshow(data)




#
#ImageReader = bioformats.formatreader.make_image_reader_class()
##reader = ImageReader()
#with bioformats.formatreader.ImageReader(path) as reader:
#    data = reader.read(path,z=0,t=0,series=0, rescale=False,XYWH= (0,0,10000,10000))
#
#data=np.reshape(data,(10000,10000,3))
#
#plt.imshow(data)

slice=4







image_info = bioformats.get_omexml_metadata(path)
omxml_data=bioformats.OMEXML(image_info)

channel_count = omxml_data.image_count

pixels = omxml_data.image(slice).Pixels
sx = pixels.SizeX
sy = pixels.SizeY




dst_ds = gdal.GetDriverByName('GTiff').Create(out_path, sy,sx, 3, gdal.GDT_Byte,['COMPRESS=LZW'])
r1=dst_ds.GetRasterBand(1)
r2=dst_ds.GetRasterBand(2)
r3=dst_ds.GetRasterBand(3)    

step=100;

with bioformats.formatreader.ImageReader(path) as reader:
    for k in range(0, sy,step):
        print(k)
        if k+step>=sy:
            step=step-(k+step-sy)-1

        data = reader.read(path,z=0,t=0,series=slice, rescale=False,XYWH= (0,k,sx,step))
        data=np.array(np.reshape(data,(step,-1,3)))
        r1.WriteArray(data[:,:,0].T,k,0)
        r2.WriteArray(data[:,:,1].T,k,0)
        r3.WriteArray(data[:,:,2].T,k,0)

dst_ds.BuildOverviews("NEAREST", [2,4,8,16,32,64])
dst_ds.FlushCache()                     # write to disk
dst_ds = None
