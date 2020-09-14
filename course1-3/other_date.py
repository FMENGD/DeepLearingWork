# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:12:52 2020

@author: FMENG
"""

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets



"""
改变数据集
"""
noisy_circles,noisy_moons,blobs,gaussian_quantiles,no_structure = load_extra_datasets()

datesets = {
        "noisy_circles":noisy_circles,
        "noisy_moons":noisy_moons,
        "blobs":blobs,
        "gaussian_quantiles":gaussian_quantiles
        
        }
dateset = "noisy_moons"

X,Y = datesets[dateset]
X,Y = X.T,Y.reshape(1,Y.shape[0])

if dateset == "blobs":
    Y = Y%2
plt.scatter(X[0,:],X[1,:],c=np.squeeze(Y),s=0,cmap=plt.cm.Spectral)
 