# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:24:15 2022

@author: Maria
"""



# Imports
import os
import glob
import sys
import numpy as np

# Project Imports
from show_patch import show_patch



# Define data path
path = 'C:/Users/Maria/Documents/GitHub/breast-cancer-classification/data/ddsm/train'
requiredInfos = np.array ([])

print ("Do you want to search masses (0) or calcifications (1)?")
decision = int (input())

# asks the user which kind of features they want see an example of
if decision == 0:
    print ("Are you looking for a specific shape (0), margin (1) or both (2)?")
    requiredFeatures = int (input())
    
    if requiredFeatures == 0 or requiredFeatures == 2:
        print ("Please insert the mass shape: ")
        # requiredInfos saves the features the user is searching for; it's size can be 1 or 2
        requiredInfos = np.append(requiredInfos, input())
    elif requiredFeatures != 1:
        # if the input isn't 0, 1 or 2 it isn't valid
        print ("Error! Invalid input!")
        sys.exit()
        
    if  int (requiredFeatures) == 1 or int (requiredFeatures) == 2:
        print ("Please insert the mass margin: ")
        requiredInfos = np.append(requiredInfos, input())

elif decision == 1:
    print ("Are you looking for a specific type (0), distribution (1) or both (2)?")
    requiredFeatures = int (input())
    
    if requiredFeatures == 0 or requiredFeatures == 2:
        print ("Please insert the calcification type: ")
        requiredInfos = np.append(requiredInfos, input())
    elif requiredFeatures != 1:
        # if the input isn't 0, 1 or 2 it isn't valid
        print ("Error! Invalid input!")
        sys.exit()
        
    if  requiredFeatures == 1 or requiredFeatures == 2:
        print ("Please insert the calcification distribution: ")
        requiredInfos = np.append(requiredInfos, input())
    
else:
    # if the intial input isn't 0 or 1 it isn't valid
    print ("Error! Invalid input!")
    sys.exit()
   
# must the path be a square?
isSquare = 1

# isRequiredExample allows the program to see if the description of a file 
# matches all the searched features (this is particularly useful when the user
# wants to search for 2 features -> lines 92-98)
isRequiredExample = np.zeros((1,np.size(requiredInfos)))

# changes to true if an example with all the required features was found -> line 99
wasFound = False

for filename in glob.glob(os.path.join(path, '*.txt')):
    file = open (filename)
    lesion_type = file.readline()
    pathology = file.readline()
    assessment = file.readline()
    subtlety = file.readline()
    info = file.readline()

    # searches for all of the features indicated by the user in each txt file; 
    # if a feature is found, a 1 is allocated to isRequiredExample; else, a 0 
    # is allocated to the same vector. This way, if the sum of the elements of 
    # the vector is equal to its size, all the relevant features were found in the file
    if ((decision == 0 and lesion_type.find("MASS") != -1) or (decision == 1 and (lesion_type.find("CALCIFICATION") != -1))):
        for i in range (0, np.size(requiredInfos)):
            if (info.find(" " + requiredInfos[i] + " ") != -1 or info.find(" " + requiredInfos[i] + "\n") != -1):
                isRequiredExample[0][i] = 1
            else:
                isRequiredExample[0][i] = 0
                break
                
        if isRequiredExample.sum() == np.size(isRequiredExample):
            wasFound = True
            arrayFilename = filename[0:len(filename) - 3] + 'npy'
            break

if wasFound:
    imgArray = np.load(arrayFilename)
    img = show_patch(imgArray, isSquare)
 
# wasFound only changes to true if a file with all the required features was found -> line 99
else:
    print ("There isn't an example with the required features in the dataset.")
