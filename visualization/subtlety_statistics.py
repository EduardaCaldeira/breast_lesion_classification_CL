# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:14:22 2022

@author: Maria
"""

import numpy as np
import glob
import os

path = 'C:/Users/Maria/Documents/GitHub/breast-cancer-classification/ddsm/train'
total = np.zeros (5)
benign = np.zeros (5)

for filename in glob.glob(os.path.join(path, '*.txt')): 
    file = open (filename)
    lesion_type = file.readline()
    pathology = file.readline()
    assessment = file.readline()
    subtlety = file.readline()
    total[int(subtlety) - 1] += 1
    if (pathology[0] == 'B'):
        benign[int(subtlety) - 1] += 1
    
for i in range (0,5):
    print ("Level", i + 1, "-> M/B:", round((total[i] - benign[i])/benign[i],4), "; level %:", round(total[i] / total.sum(),4))
