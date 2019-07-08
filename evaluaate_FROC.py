import gdal
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import os
import sys
import pandas
from scipy.interpolate import griddata

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import median_filter

from skimage.morphology import disk,dilation
from skimage.feature import peak_local_max

def imread_gdal_mask(data_name,level):
    
    level=level-1

    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        
    rOverview = rOverview.ReadAsArray(0,0, rOverview.XSize, rOverview.YSize) 
    
    return rOverview


def imread_gdal_mask(data_name,level):
    
    level=level-1

    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand(1)
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

   
def computeEvaluationMask(maskDIR, resolution, level):
    """Computes the evaluation mask.
    
    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made
        
    Returns:
        evaluation_mask
    """
#    slide = openslide.open_slide(maskDIR)
#    dims = slide.level_dimensions[level]
#    pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
#    pixelarray = np.array(slide.read_region((0,0), level, dims))
#    
    
    
    pixelarray=(imread_gdal_mask(maskDIR,level)>0)*255
#    print(np.sum(pixelarray)/255)
    distance = nd.distance_transform_edt(255 - pixelarray)
    Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity = 2) 
    return evaluation_mask
    
    
def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)
    
    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object 
        should be less than 275µm to be considered as ITC (Each pixel is 
        0.243µm*0.243µm in level 0). Therefore the major axis of the object 
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.
        
    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made
        
    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)    
    properties = measure.regionprops(evaluation_mask,coordinates='rc')
    Isolated_Tumor_Cells = [] 
    threshold = 275/(resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i+1)
    return Isolated_Tumor_Cells


def readCSVContent(csvDIR):
    """Reads the data inside CSV file
    
    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image
        
    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
#    Xcorr, Ycorr, Probs = ([] for i in range(3))
#    csv_lines = open(csvDIR,"r").readlines()
#    for i in range(len(csv_lines)):
#        line = csv_lines[i]
#        elems = line.rstrip().split(',')
#        Probs.append(float(elems[0]))
#        Xcorr.append(int(elems[1]))
#        Ycorr.append(int(elems[2]))
#    return Probs, Xcorr, Ycorr
    
    ddd= pandas.read_csv(csvDIR)
    Probs=ddd['Confidance'].values.tolist()
    Xcorr=np.round(ddd['X coordinate'].values).tolist()
    Ycorr=np.round(ddd['Y coordinate'].values).tolist()
    
    Xcorr = [int(num)  for num in Xcorr]
    Ycorr = [int(num)  for num in Ycorr]
    return Probs, Xcorr, Ycorr
         
def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
    """Generates true positive and false positive stats for the analyzed image
    
    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made
         
    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections
        
        TP_probs:   A list containing the probabilities of the True positive detections
        
        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)
        
        detection_summary:   A python dictionary object with keys that are the labels 
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate]. 
        Lesions that are missed by the algorithm have an empty value.
        
        FP_summary:   A python dictionary object with keys that represent the 
        false positive finding number and values that contain detection 
        details [confidence score, X-coordinate, Y-coordinate]. 
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = [] 
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}  
    FP_summary = {}
    for i in range(1,max_label+1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []        
     
    FP_counter = 0       
    if (is_tumor):
        for i in range(0,len(Xcorr)):
            HittedLabel = evaluation_mask[int(Ycorr[i]), int(Xcorr[i])]#pow(2, level)
            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter+=1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i]>TP_probs[HittedLabel-1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel-1] = Probs[i]                                     
    else:
        for i in range(0,len(Xcorr)):
            FP_probs.append(Probs[i]) 
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]] 
            FP_counter+=1
            
    num_of_tumors = max_label - len(Isolated_Tumor_Cells);                             
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary
 
 
def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve
    
    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image
         
    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """
    
    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist] 
    
    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())    
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs)/float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs)/float(sum(FROC_data[3]))      
    return  total_FPs, total_sensitivity
   
   
def plotFROC(total_FPs, total_sensitivity):
    """Plots the FROC curve
    
    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
         
    Returns:
        -
    """    
    fig = plt.figure()
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)  
    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(total_FPs, total_sensitivity, '-', color='#000000')    
    plt.show()       
def makeGaussian(sig = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    size=2*np.ceil(2*sig)+1
    
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sig**2))      
    






def froc_eval(g,t,sig,name,data_type):
    
    mask_folder = "..\\dataset\\"+ data_type +"_cam\\mask"
    result_folder = "..\\results\\"+ name +"\\"+data_type
    
    result_file_list = []
    result_file_list += [each for each in os.listdir(result_folder) if each.endswith('.tif')]
    
#    result_file_list= [result_file_list[i] for i in [1,0,20,43,61,66,81,99,102,111,10,71]]
#    result_file_list= [result_file_list[i] for i in [1]]
    
#    if data_type=='valid':
#        result_file_list= [result_file_list[i] for i in [2,0,5,7,9,11,13,15,17]]
#    else:
#        result_file_list= [result_file_list[i] for i in np.arange(0,110,3)]
    
    
    EVALUATION_MASK_LEVEL = 5 # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243 # pixel resolution at level 0
    
    FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    
    caseNum = 0    
    for case in result_file_list:
        print(str(g)+'  '+str(t)+'  '+str(sig)+'  '+'Evaluating Performance on image:', case[0:-4])
        sys.stdout.flush()
#        csvDIR = os.path.join(result_folder, case)
#        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)
        
        result_img_folder="..\\results\\"+ name +"\\" + data_type
        r_n= os.path.join(result_img_folder, case[0:-4]) + '.tif'
        res_img=imread_gdal_mask_small(r_n,EVALUATION_MASK_LEVEL-3)
        
        
        
        
        r=int(((2*np.ceil(2*sig)+1)-1)/2)
        
        Probs, Xcorr, Ycorr=[],[],[]
        res_img=median_filter(res_img,5)
        res_img=gaussian_filter(res_img,g)
        res_img[:r,:]=0
        res_img[-r:,:]=0
        res_img[:,-r:]=0
        res_img[:,:r]=0
        
#        res_0=res_img
#        
#        cont=1
#        while cont:
#            ind=np.argmax(res_img,axis=None)
#            ind = np.unravel_index(ind, res_img.shape)
#            v=res_0[ind[0],ind[1]]
#            if v>t:
#                Probs.append(v)
#                Xcorr.append(ind[1])
#                Ycorr.append(ind[0])
#                rem=np.ones(np.shape(res_img),dtype=np.float32)
#                di=makeGaussian(sig)
#                rem[ind[0]-r:ind[0]-r+np.shape(di)[0],ind[1]-r:ind[1]-r+np.shape(di)[1]]=1-di
#
#                res_img=res_img*rem
#                
#            else:
#                cont=0
        
        
        
        tmp =peak_local_max(res_img, min_distance=int(sig), threshold_abs=t)
        Ycorr=list(tmp[:,0])
        Xcorr=list(tmp[:,1])
        Probs=list(res_img[Ycorr,Xcorr])
        
      
#        plt.imshow(res_img)
#        plt.plot(Xcorr, Ycorr,'*')
    
                
        

        is_tumor = case[0:5] == 'tumor'    
        if (is_tumor):
            maskDIR = os.path.join(mask_folder, case[0:-4]) + '_mask.tif'
            evaluation_mask = computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            
#            plt.figure()    
#            plt.imshow(evaluation_mask)
#            plt.plot(Xcorr, Ycorr,'*')
#            a=fdfdfdfd
            
        else:
            evaluation_mask = 0
            ITC_labels = []
#            
        
#        a=fsddf
#           
        FROC_data[0][caseNum] = case
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case
        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels, EVALUATION_MASK_LEVEL)
        caseNum += 1

    # Compute FROC curve 
    total_FPs, total_sensitivity = computeFROC(FROC_data)
#    sfsdfd=fdfsdfsd
    
    # plot FROC curve
#    plotFROC(total_FPs, total_sensitivity)
    if len(total_FPs)>2:
        results0 = griddata(total_FPs, total_sensitivity,[0.25,0.5,1,2,4,8])
        print(results0 )
        results=np.array(results0)
        if np.sum(np.isnan(results))<=6:
            results[np.isnan(results)]=results[np.isnan(results)==0][-1]
    else:
        results=np.ones(6)*np.nan
    
    print(results )
    froc=np.mean(results)
    print(froc)
    return results,froc




if __name__ == "__main__":

    name='1s_no_pixel'
    
    gs=np.linspace(3,50,4)
    ts=[0.3]
    sigs=np.linspace(30,150,4)
    
#    gs=[10]
#    ts=[0.5]
#    sigs=[80]
    
    
    results_FROC=np.zeros((len(gs),len(ts),len(sigs)))
    results_FROC_all=np.zeros((len(gs),len(ts),len(sigs),6))
    
    ggs=np.zeros((len(gs),len(ts),len(sigs)))
    tts=np.zeros((len(gs),len(ts),len(sigs)))
    sigsigs=np.zeros((len(gs),len(ts),len(sigs)))
    
    for k in range(len(gs)):
        for kk in range(len(ts)):
            for kkk in range(len(sigs)):
                g=gs[k]
                t=ts[kk]
                sig=sigs[kkk]
                
                
                ggs[k,kk,kkk]=g
                tts[k,kk,kkk]=t
                sigsigs[k,kk,kkk]=sig
                
                results,froc=froc_eval(g,t,sig,name,'valid')
                
                
                results_FROC[k,kk,kkk]=froc
                results_FROC_all[k,kk,kkk,:]=results 
       
        
    ind=np.argmax(results_FROC,axis=None)
    ind = np.unravel_index(ind, results_FROC.shape)
    
    g=ggs[ind[0],ind[1],ind[2]]
    t=tts[ind[0],ind[1],ind[2]]
    sig=sigsigs[ind[0],ind[1],ind[2]]    
        
    results,froc=froc_eval(g,t,sig,name,'test')
    
    np.save('../results/' +name+ '/results_froc.npy',froc)
    np.save('../results/' +name+ '/results_froc_all.npy',results)
    np.save('../results/' +name+ '/froc_valid.npy',results_FROC)                 
    np.save('../results/' +name+ '/ggs.npy',ggs)     
    np.save('../results/' +name+ '/tts.npy',tts) 
    np.save('../results/' +name+ '/sigsigs.npy',sigsigs)  