# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:57:12 2018

@author: tomas
"""
import numpy as np

a=np.array([[0,0,1],[1,0,0],[0,1,0]])
p=np.random.rand(3,3)
p=np.array([[0,0,20],[20,0,0],[0,1,0]])

pozice=np.where(a)

pp=p[pozice]

pp=pp/np.sum(pp)

kde=np.random.choice(len(pozice[0]),size=2, replace=False, p=pp)


q=pozice[0][kde]
qq=pozice[1][kde]


