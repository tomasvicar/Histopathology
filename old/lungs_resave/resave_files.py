import javabridge
import bioformats
import numpy as np
import os
import gdal
from read_labeling2 import read_lbl2
import gc
import cv2


def write_gdal(data,name):
    
    dst_ds = gdal.GetDriverByName('GTiff').Create(name, np.shape(data)[1], np.shape(data)[0], 3, gdal.GDT_Byte,['COMPRESS=LZW'])
    
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



def czi2tif(nameczi,nametif,slice):
    
    

    image_info = bioformats.get_omexml_metadata(nameczi)
    omxml_data=bioformats.OMEXML(image_info)
     
    pixels = omxml_data.image(slice).Pixels
    sx = pixels.SizeX
    sy = pixels.SizeY



    
    dst_ds = gdal.GetDriverByName('GTiff').Create(nametif, sx,sy, 3, gdal.GDT_Byte,['COMPRESS=LZW','BIGTIFF=YES','TILED=YES'])
    r1=dst_ds.GetRasterBand(1)
    r2=dst_ds.GetRasterBand(2)
    r3=dst_ds.GetRasterBand(3)    
    
    step=int(np.round(sy/4));
    
    with bioformats.formatreader.ImageReader(nameczi) as reader:
        for k in range(0, sy,step):
#            print(k)
            if k+step>=sy:
                step=step-(k+step-sy)-1
    
            if step>0:
                data = reader.read(nameczi,z=0,t=0,series=slice, rescale=False,XYWH= (0,k,sx,step))
                data=np.array(np.reshape(data,(step,-1,3)))
                r1.WriteArray(data[:,:,0],0,k)
                r2.WriteArray(data[:,:,1],0,k)
                r3.WriteArray(data[:,:,2],0,k)
    
    dst_ds.BuildOverviews("NEAREST", [2,4,8,16,32,64])
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None

use = [
        [2,3,4],
        [],
        [1],
        [],
        [1],
        [],
        [],
        [1],
        [1,3,4],
        [],
        [],
        [],
        [1],
        [],
        [],
        [1],
        [],
        [],
        [1,3],
        [],
        [],
        [],
        [],
        [],
        [1],
        [],
        [],
        [1],
        [1,3],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        []
        ]



javabridge.start_vm(class_path=bioformats.JARS)


folder=r'Novemember Labels'
folder_save=r'tifs'
tumor=1


#folder=r'D:\MPI_CBG\data_plice\data\normal'
#folder_save=r'D:\MPI_CBG\data_plice\data\normal_tif'
#tumor=0


if tumor:
    
    label_dict=	{'??': 1, 'solid':2,'diffuse':3}
    
    
    files=os.listdir(folder)
    files_labeling = [i for i in files if i.endswith('.labeling')]
    
else:
    
    files=os.listdir(folder)
    files_labeling = [i for i in files if i.endswith('.czi')]
    

if tumor:
    pom_name='tumor'
else:
    pom_name='normal'

files_labeling=files_labeling[32+10:]
#files_labeling=files_labeling[0:17]
#files_labeling=files_labeling[16:]
iq=-1;
for image_name in files_labeling:
    iq=iq+1
    
    print(image_name)
    
    
    img_path=folder + '\\' + image_name[0:16] + '.czi'
    
    
#    ########xremove
#    if image_name!='2017_11_30__0034-4.czi.labeling':
#        continue  
#    ##########
    
    img_path_save=folder_save + '\\' + image_name +'_' + pom_name + '.tif'
    
    if tumor:
        lbl_path = folder + '\\' + image_name
        
        lbl_path_save=folder_save + '\\' + image_name +'_' + pom_name + '_mask.tif'
        
    
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
            print(x,y)
            if (lbl_size[0]==x) and (lbl_size[1]==y):
                ii=i
#                break
        if ii==0:
            raise NameError('chanel not exist')
            
#            
#        ########xremove
#        ii=17
#        ##########
#        lbl=cv2.resize(lbl,(9442,4383), interpolation = cv2.INTER_NEAREST)
    
        lbl=cv2.resize(lbl,None,fx=2, fy=2, interpolation = cv2.INTER_NEAREST)
        write_gdal(lbl,lbl_path_save)
        
        del lbl
        gc.collect()
        
#        img_l = bioformats.load_image(img_path,z=0,t=0,series=ii-1, rescale=False)
        ######xremove
#        print(np.shape(img_l))
        ##########
    
#        write_gdal(img_l,img_path_save)
        
        czi2tif(img_path,img_path_save,ii-1)

        
#        del img_l
        gc.collect()
        
        
        
    else:
        
        image_info = bioformats.get_omexml_metadata(img_path)
        omxml_data=bioformats.OMEXML(image_info)
        
        channel_count = omxml_data.image_count
        ii=0
        sizes=[]
        m=[]
        xy_last=0
        for i in range(channel_count):
            pixels = omxml_data.image(i).Pixels
            y = pixels.SizeX
            x = pixels.SizeY
#            print(x,y)
            sizes.append(x*y)
            xy=x*y
            if xy>xy_last:
                m.append(i);
            xy_last=xy
            
        ind=np.argsort(sizes)
        
        for k in use[iq]:
            ii=m[k-1]
            
            img_path_save=folder + '_tif\\' + image_name +'_' + pom_name + '_' + str(k) + '.tif'
            
            czi2tif(img_path,img_path_save,ii)
        
#        img_l = bioformats.load_image(img_path,z=0,t=0,series=ii, rescale=False)
#        write_gdal(img_l,img_path_save)

#        a=hgggdfg   
            
#        a=1
        


