
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import os
from skimage.io import imsave
import cv2
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.color import rgb2lab
import gc
import math
from skimage.io import imread
import gdal

from numpy import linalg as LA

from mpl_toolkits.mplot3d import Axes3D




def imread_gdal(data_name,level):
    

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





#img_path=r'D:\MPI_CBG\data_plice\dataset\train\data\2017_10_24__0204.czi_normal_1.tif'
#fg_path=r'D:\MPI_CBG\data_plice\dataset\train\fg\2017_10_24__0204.czi_normal_1.tif'
#logarithm=1
#name_mac='HE_mac_plice_ref.npy'
#name_vah='HE_vah_plice_ref.npy'
#lvl=2
#
#
#img_path=r'D:\MPI_CBG\data_plice\dataset\train\data\2017_10_24__0204.czi_normal_1.tif'
#fg_path=r'D:\MPI_CBG\data_plice\dataset\train\fg\2017_10_24__0204.czi_normal_1.tif'
#logarithm=0
#name_mac='HE_mac_plice_nolog_ref.npy'
#name_vah='HE_vah_plice_nolog_ref.npy'
#lvl=2
#
#
#cam je na lvlu x 
#
img_path=r'D:\MPI_CBG\camelyon16\dataset\train\data\tumor_032.tif'
fg_path=r'D:\MPI_CBG\camelyon16\dataset\train\fg\tumor_032.tif'
logarithm=1
name_mac='HE_mac_cam_ref.npy'
name_vah='HE_vah_cam_ref.npy'
lvl=3
##
#
##
#img_path=r'D:\MPI_CBG\camelyon16\dataset\train\data\tumor_032.tif'
#fg_path=r'D:\MPI_CBG\camelyon16\dataset\train\fg\tumor_032.tif'
#logarithm=0
#name_mac='HE_mac_cam_nolog_ref.npy'
#name_vah='HE_vah_cam_nolog_ref.npy'
#lvl=3













I =imread_gdal(img_path,lvl)

#plt.imshow(I)
#a=fsdfsdfsd

maska=imread(fg_path)>0
    

beta=0.15
alpha=0.99

lam=0.1


#Ip=rgb2lab(I/255)[:,:,0]
#pouzit=Ip>0.9
#pouzit=np.reshape(pouzit,(-1,1))

I=np.reshape(I,(-1,3))
maska=np.reshape(maska,(-1,1))

I=I[np.nonzero(maska)[0],:]


I=I.astype(np.float64)
I=(I+1)/255

if logarithm:
    OD=-np.log10(I)
else:
    OD=1-I




HEE=[]
for k in range(10):
    
    hotovo=0
    while not hotovo:
        r=np.random.choice(np.shape(OD)[0],10000)
        
        Ihat=I[r,:]
        
        ODhat=OD[r,:]
        
        # eigenvectors of cov in OD space (orthogonal as cov symmetric)
        try:
            w, V = LA.eigh(np.cov(ODhat, rowvar=False))
            
            hotovo=1
        except:
            print('chybka')
            pass
#    u, s, vh= LA.svd(ODhat)
    
    # the two principle eigenvectors
    V = V[:, [1, 2]]
    # make sure vectors are pointing the right way
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    # project on this basis.
    That = np.dot(ODhat, V)
    
    # angular coordinates with repect to the prinicple, orthogonal eigenvectors
    phi = np.arctan2(That[:, 1], That[:, 0])
    # min and max angles
    minPhi = np.percentile(phi, 100 - alpha)
    maxPhi = np.percentile(phi, alpha)
    
    # the two principle colors
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    
    # order of H and E.
    # H first row.
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    
    
    HE=HE / LA.norm(HE, axis=1)[:, None]
    
    HEE.append(HE)
    
    
HE=np.median(np.stack(HEE,axis=2), axis=2)  
HE_mac=HE






from rpy2.robjects.packages import importr
utils = importr('utils')
utils.chooseCRANmirror(ind=1)

packnames = ('NNLM','hexbin')
from rpy2.robjects.vectors import StrVector
utils.install_packages(StrVector(packnames))


NNLM = importr('NNLM')


import rpy2.robjects as robjects
from rpy2.robjects import FloatVector

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()









HEE=[]
for k in range(10):
    
    hotovo=0

    r=np.random.choice(np.shape(OD)[0],10000)
    
    Ihat=I[r,:]
    
    ODhat=OD[r,:]
    
    
    decomp=NNLM.nnmf(ODhat.T, k = 2, alpha = [0,0,0], beta = [0,0,lam], method = "scd", loss = "mse")


    W=decomp.rx2('W')
    W = np.array(W)
    
    print(W)
    
    
    dictionary=W
    dictionary=dictionary.T
    
    
    
    
    

    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]
    
    HE=dictionary
    HE=HE / LA.norm(HE, axis=1)[:, None]
    
    HEE.append(HE)
    
    
HE=np.median(np.stack(HEE,axis=2), axis=2)  
HE_vah=HE





 
ax = plt.axes(projection='3d')

a=ODhat[:,0]
b=ODhat[:,1]
c=ODhat[:,2]
    

#a=Ihat[:,0]
#b=Ihat[:,1]
#c=Ihat[:,2]


Ihat[Ihat>1]=1
ax.scatter(a,b,c, c= Ihat)  
ax.plot([0,HE_vah[0,0]],[0,HE_vah[0,1]],[0,HE_vah[0,2]],'r')
ax.plot([0,HE_vah[1,0]],[0,HE_vah[1,1]],[0,HE_vah[1,2]],'r')
ax.plot([0,HE_mac[0,0]],[0,HE_mac[0,1]],[0,HE_mac[0,2]],'b')
ax.plot([0,HE_mac[1,0]],[0,HE_mac[1,1]],[0,HE_mac[1,2]],'b')
plt.show()






np.save(name_mac,HE_mac)
np.save(name_vah,HE_vah)

    