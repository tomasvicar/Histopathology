import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from skimage.io import imsave
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.color import rgb2lab



from numpy import linalg as LA

from mpl_toolkits.mplot3d import Axes3D


def get_stain_macenko_vah(I,maska,he_mac_ref_name,he_vah_ref_name,logarithm=True):

    

    
    
#    beta=0.15
    alpha=0.99
    
    lam=0.1
    
    
    I=np.reshape(I,(-1,3))
    maska=np.reshape(maska,(-1,1))
    
    I=I[np.nonzero(maska)[0],:]
    
    
    I=I.astype(np.float64)
    I=(I+1)/255
    if logarithm:
        OD=-np.log10(I)
    else:
        OD=1-I
#    pouzit=np.bitwise_and(np.bitwise_and(OD[:,1]>beta,OD[:,2]>beta),OD[:,0]>beta)
    
#    OD=OD[np.nonzero(pouzit)[0],:]
#    Ih=I[np.nonzero(pouzit)[0],:]
    
    
    HEE=[]
    for k in range(5):
        
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
    for k in range(5):
        
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
        
        
        
    
    #    dictionary = dict_learning(ODhat, 2, lam)[1].T
        
        
        
        
    #    model = NMF(2,l1_ratio=1,alpha=0.1)
    #    
    #    dictionary=model.fit_transform(ODhat.T)
    #    dictionary=dictionary.T
    #    
        
        
    #    snmf = nimfa.Snmf(ODhat.T, seed="random_vcol", rank=2, max_iter=100, version='r', eta=0, beta=0.1, i_conv=10, w_min_change=0)
    ##    
    #    snmf_fit=snmf.factorize()
    ##
    #    dictionary=snmf_fit.basis()
    ##    dictionary=snmf_fit.coef()
    #
    #    dictionary=dictionary.T
        
        
        
        
    
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
    ax.scatter(a,b,c, c= Ihat )  
    ax.plot([0,HE_vah[0,0]],[0,HE_vah[0,1]],[0,HE_vah[0,2]],'r')
    ax.plot([0,HE_vah[1,0]],[0,HE_vah[1,1]],[0,HE_vah[1,2]],'r')
    ax.plot([0,HE_mac[0,0]],[0,HE_mac[0,1]],[0,HE_mac[0,2]],'b')
    ax.plot([0,HE_mac[1,0]],[0,HE_mac[1,1]],[0,HE_mac[1,2]],'b')
    plt.show()



    HE_mac_inv=np.linalg.pinv(HE_mac)
    HE_vah_inv=np.linalg.pinv(HE_vah)


    
    
    data_v=np.reshape(I,(-1,3))
    if logarithm:
        data_v=-np.log10((data_v))
    else:
        data_v=1-(data_v)
    data_t=np.matmul(data_v,HE_mac_inv)
    
    mac_max=[np.percentile(data_t[:,0],alpha),np.percentile(data_t[:,1],alpha)]
    
    mac_mean=[np.mean(data_t[:,0]),np.mean(data_t[:,1])]

    mac_std=[np.std(data_t[:,0]),np.std(data_t[:,1])]

     
    

    data_v=np.reshape(I,(-1,3))
    if logarithm:
        data_v=-np.log10((data_v))
    else:
        data_v=1-(data_v)
    data_t=np.matmul(data_v,HE_vah_inv)
    
    vah_max=[np.percentile(data_t[:,0],alpha),np.percentile(data_t[:,1],alpha)]
    
    vah_mean=[np.mean(data_t[:,0]),np.mean(data_t[:,1])]

    vah_std=[np.std(data_t[:,0]),np.std(data_t[:,1])]
    
    
    
    
    
    

    HE_mac_s=np.load(he_mac_ref_name)
    HE_vah_s=np.load(he_vah_ref_name)

        
        
    HE_mac_inv_s=np.linalg.pinv(HE_mac_s)
    HE_vah_inv_s=np.linalg.pinv(HE_vah_s)
    
    data_v=np.reshape(I,(-1,3))
    if logarithm:
        data_v=-np.log10((data_v))
    else:
        data_v=1-(data_v)
    data_t=np.matmul(data_v,HE_mac_inv_s)
    
    mac_max_s=[np.percentile(data_t[:,0],alpha),np.percentile(data_t[:,1],alpha)]
    
    mac_mean_s=[np.mean(data_t[:,0]),np.mean(data_t[:,1])]

    mac_std_s=[np.std(data_t[:,0]),np.std(data_t[:,1])]
    
    
    data_v=np.reshape(I,(-1,3))
    if logarithm:
        data_v=-np.log10((data_v))
    else:
        data_v=1-(data_v)
    data_t=np.matmul(data_v,HE_vah_inv_s)
    
    vah_max_s=[np.percentile(data_t[:,0],alpha),np.percentile(data_t[:,1],alpha)]
    
    vah_mean_s=[np.mean(data_t[:,0]),np.mean(data_t[:,1])]

    vah_std_s=[np.std(data_t[:,0]),np.std(data_t[:,1])]
    

    return mac_max,mac_mean,mac_std,HE_mac_inv,HE_mac,vah_max,vah_mean,vah_std,HE_vah_inv,HE_vah,mac_max_s,mac_mean_s,mac_std_s,HE_mac_inv_s,HE_mac_s,vah_max_s,vah_mean_s,vah_std_s,HE_vah_inv_s,HE_vah_s


