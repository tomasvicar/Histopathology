import cv2
import numpy as np

def img_with_cont(img,mask,c=(0,0,255),th=1):
    
    mask=np.uint8((np.squeeze(mask)>0)*255)
    mask=np.squeeze(mask)
    
    
    img=np.array(img,dtype=np.uint8)
    
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, c, th)
    return img