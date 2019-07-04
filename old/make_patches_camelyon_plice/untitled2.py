# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 19:41:16 2018

@author: Tom
"""
import numpy as np
import gdal
name=r'D:\MPI_CBG\data_plice\dataset\test\mask\2017_11_30__0033-2.czi.labeling_tumor_mask.tif'

def imread_gdal_mask(data_name,level):
    
    level=level-1

    gdalObj = gdal.Open(data_name)
    

    
    rOverview = gdalObj.GetRasterBand (1)
    if level>=0:
        rOverview = rOverview.GetOverview(level)
        
    rOverview = rOverview.ReadAsArray(0,0, rOverview.XSize, rOverview.YSize) 
    
    return rOverview

a=imread_gdal_mask(name,0)