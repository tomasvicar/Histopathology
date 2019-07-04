import os
import gdal
import cv2
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import binary_fill_holes as imfill
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import regionprops

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import median_filter
import time
from skimage.io import imread



def img_with_cont(img,mask,c=(0,0,255)):
    
    mask=np.uint8((np.squeeze(mask)>0)*255)
    mask=np.squeeze(mask)
    
    
    img=np.array(img,dtype=np.uint8)
    
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, c, 10)
    return img



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


def imread_gdal_mask(data_name,level):
    
    level=level-1

    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        
    rOverview = rOverview.ReadAsArray(0,0, rOverview.XSize, rOverview.YSize) 
    
    return rOverview



def imread_gdal_mask_small(data_name,level):
    
    level=level-1

    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        
    rOverview = rOverview.ReadAsArray(0,0, int(np.ceil(rOverview.XSize/2)), int(np.ceil(rOverview.YSize/2))) 
    
    return rOverview



def get_features(r,m):
    f=np.array([],dtype=np.float64)
    m=m.astype(np.uint8())
    props=regionprops(label_image=m, intensity_image=r,coordinates='rc')
    f=np.append(f,props[0].eccentricity)
    f=np.append(f,props[0].solidity)
    f=np.append(f,props[0].area)
    f=np.append(f,props[0].max_intensity)
    f=np.append(f,props[0].mean_intensity)
    f=np.append(f,props[0].min_intensity)
    f=np.append(f,props[0].weighted_moments_central)

    
    return f


features=[]
gt=[]
img_names=[]
img_number=[]
centroids=[]
for data_type in ['valid','test']:
    f=np.zeros(shape=(0,5),dtype=np.float64)
    y=np.array([],dtype=np.float64)
    
    
    
    lvl=3
    
    model_name='1s_no_unet'
    save_folder='..\\results\\' +model_name +'\\'+data_type
    data_path=r'Z:\CELL_MUNI\histolky\dataset\\'+  data_type +'_cam\data'
    
    file_names=[]
    for root, dirs, files in os.walk(data_path):
        for name in files:
            if name.endswith(".tif"):
                file_names.append(root+'\\'+name)
                
      
#    i=0
#    file_names=file_names[i:i+1]   
                
                
                
    ff=[] 
    y=[]            
#    file_names=file_names[10:]  
    img_counter=-1
    for img_filename in file_names:
        img_counter+=1
        
        print(str(img_counter) + '/'+ str(len(img_filename)) + '     ' + img_filename)
        
        

        fg_name=img_filename
        
        fg_name=img_filename
        fg_name=fg_name.replace('\\data\\','\\fg\\')
        
        lbl_name=fg_name
        lbl_name=lbl_name.replace('\\fg\\','\\mask\\')
        lbl_name=lbl_name.replace('.tif','_mask.tif')
        
        tmp=img_filename.split('\\')
        tmp=tmp[-1]
        result_name=save_folder +'\\'+ tmp
        
        if not os.path.isfile(result_name):
            continue
            
        
        result=imread_gdal_mask_small(result_name,lvl)
        data=imread_gdal(img_filename,lvl+2)
        if 'tumor_' in lbl_name:
            gt=imread_gdal_mask(lbl_name,lvl+2)>0
        else:
            gt=np.zeros(np.shape(result),dtype=np.bool)
            
        
        
        start_time = time.time()
        result=median_filter(result,5)
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        result=gaussian_filter(result,2)
        print("--- %s seconds ---" % (time.time() - start_time))
#        
#        plt.imshow(result)
#        plt.show()
#        dsfsd=sdfsdfds
            
        rui8=result
        rui8=((rui8-0.85)/0.15)*255
        rui8[rui8<0]=0
        rui8[rui8>255]=255
        rui8=rui8.astype(np.uint8)

        mser = cv2.MSER_create(_delta = 5,_min_area=200,_max_area = int(np.floor(np.sum(result>0)*0.1)),_max_variation = 0.2,_min_diversity = 0.2)
        regions,bbs = mser.detectRegions(rui8)
        
#        new_regions=[]
#        new_bbs=[]
#        last_bb=np.array([0,0,1,1])
#        for i,bb in enumerate(bbs):
#            lu_corner_x=np.max((last_bb[0],bb[0]))
#            lu_corner_y=np.max((last_bb[1],bb[1]))
#            rd_corner_x=np.min((last_bb[0]+last_bb[2],bb[0]+bb[2]))
#            rd_corner_y=np.min((last_bb[1]+last_bb[3],bb[1]+bb[3]))
#            if (rd_corner_x-lu_corner_x)>0 and rd_corner_y-lu_corner_y>0:
#                intersection=(rd_corner_x-lu_corner_x)*(rd_corner_y-lu_corner_y)
#            else:
#                intersection=0
#            union=last_bb[2]*last_bb[3]+bb[2]*bb[3]
#            if intersection/union<0.3:
#                last_bb=bb
#                new_regions.append(regions[i])
#                new_bbs.append(bb)
#
#        regions=new_regions
#        bbs=new_bbs
        
        remove=[]
        k=-1
        kk=-1
        while k < np.shape(bbs)[0]-1:
            k+=1
            bb1=bbs[k]
            if np.sum(k==remove)>0:
                    continue
            kk=k
            while kk < np.shape(bbs)[0]-1:
                if np.sum(kk==remove)>0:
                    continue
                    
                
                kk+=1
                bb2=bbs[kk]
                
                lu_corner_x=np.max((bb1[0],bb2[0]))
                lu_corner_y=np.max((bb1[1],bb2[1]))
                rd_corner_x=np.min((bb1[0]+bb1[2],bb2[0]+bb2[2]))
                rd_corner_y=np.min((bb1[1]+bb1[3],bb2[1]+bb2[3]))
                if (rd_corner_x-lu_corner_x)>0 and rd_corner_y-lu_corner_y>0:
                    intersection=(rd_corner_x-lu_corner_x)*(rd_corner_y-lu_corner_y)
                else:
                    intersection=0
                union=bb1[2]*bb1[3]+bb2[2]*bb2[3]
                if intersection/union>0.5:
                    remove.append(kk)

        
        bbs=np.delete(bbs,remove,axis=0)
        
        regions = [x for i,x in enumerate(regions) if i not in remove]
#        del regions[]
#        regions=np.delete(regions,remove)
        
        
        
        regions_img=np.zeros(np.shape(rui8))
        for i in range(len(regions)):
#            print(str(i) + '//' + str(len(regions)))
            bb_mask=np.zeros(bbs[i,2:4])
            a=bbs[i,0]
            b=bbs[i,1]
            cv2.fillPoly(bb_mask,pts=[np.stack([regions[i][:,1]-b,regions[i][:,0]-a],axis=1)],color=[1,1,1])
            
            bb_mask = imfill(bb_mask)
            
            bb_gray=result[bbs[i,1]:(bbs[i,1]+bbs[i,3]),bbs[i,0]:(bbs[i,0]+bbs[i,2])].T
            
            bb_gt=gt[bbs[i,1]:(bbs[i,1]+bbs[i,3]),bbs[i,0]:(bbs[i,0]+bbs[i,2])].T
            
            f=get_features(bb_gray,bb_mask)
            
            ff.append(f)
            
            c=np.array(center_of_mass(bb_gray,bb_mask>0,np.arange(0,1)))
            label=bb_gt[c[1],c[0]]
            y.append(label)
            
            
#            
#            if np.sum(bb_mask[bb_mask>0])>0.4*np.sum(bb_mask>0) and centroid:
            
                
            
#            plt.figure()
#            plt.imshow(bb_mask)
#            plt.show()
#            plt.figure()
#            plt.imshow(bb_gray)
#            plt.show()
            
            regions_img[bbs[i,1]:(bbs[i,1]+bbs[i,3]),bbs[i,0]:(bbs[i,0]+bbs[i,2])]=regions_img[bbs[i,1]:(bbs[i,1]+bbs[i,3]),bbs[i,0]:(bbs[i,0]+bbs[i,2])]+bb_mask.T
            
            
#        plt.figure()   
#        plt.imshow(regions_img)
#        plt.show()
#        
#        plt.figure()
#        plt.imshow(result)
#        plt.show()
            
#        img_c=img_with_cont(255*np.stack((result,result,result),axis=2),regions_img>0,c=(0,0,255));
#        plt.imshow(img_c)
            
#        dsfsd=sdfsdfds
            
#            c=np.array(center_of_mass(result,regions_img>0,np.arange(0,1)))
#            lablel=gt[c[1],c[0]]
#            f=get_features(region_img_tmp,result)
            
            
            
        
        
        
        
        
        