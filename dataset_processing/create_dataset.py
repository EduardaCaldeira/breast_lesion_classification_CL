# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:04:47 2022

@author: Maria
"""



# Imports
import glob
import os
import sys
import tqdm
import numpy as np
import pandas as pd
from PIL import Image

# Project Imports
from auxiliary import get_mask_patch
from encode_attributes2 import encode_attributes
from get_label import get_label

def to_raw(string):
    return fr"{string}"
# Function: Create data set
def create_dataset(path='C:/Users/david/OneDrive/Documentos/GitHub/breast-cancer-classification'):
    # If something fails you may need to set your project path (uncomment if needed)
    # It is better to set PATH as the current working directory (to do this we use "os.getcwd()"")
    # path = os.getcwd()
    # path = 
    
    # Append current working directory to PATH to export stuff outside this folder
    if path not in sys.path:
        sys.path.append(path)
    
    
    
    # All the images should have the same size
    IMG_SIZE = 224
    
    
    # Array that will allow us to distiguish training, validation and test examples
    extraPath = np.array (['/train', '/val', '/test'])

    # Go through each example in the array
    for i in tqdm.tqdm(range(0, extraPath.size)):
        # is increased when a new image is considered in order to name the new 
        # files properly
        countMasses = 0
        countCalc = 0
        countMC = 0
        
        # Dictionaries that will be used to export the .csv files
        yM = {'imgName': []}
        attrM = {'imgName': [],
                     'label': [],
                     'attr1': [],
                     'attr2': [],
                     'subtlety': []}
        
        yC = {'imgName': []}
        attrC = {'imgName': [],
                     'label': [],
                     'attr1': [],
                     'attr2': [],
                     'subtlety': []}
        
        yMC = {'imgName': []}
        attrMC = {'imgName': [],
                     'label': [],
                     'attr1': [],
                     'attr2': [],
                     'attr3': [],
                     'attr4': [],
                     'subtlety': []}
       
        # Path that leads to the raw data in each subset (train, val and test)
        rawPath = path + '/data/ddsm' + extraPath[i]
        # rawPath = 'data/ddsm' + extraPath[i]
        
        # Creates the path that will lead to the processed data in each subset
        procPathMasses = path + '/data/processed/masses' + extraPath[i]
        # procPathMasses = 'data/processed/masses' + extraPath[i]
        
        procPathCalc = path + '/data/processed/calcifications' + extraPath[i]
        # procPathCalc = 'data/processed/calcifications' + extraPath[i]
        
        # some lesions are classified as both masses and calcifications; these lesions
        # will be kept in a separate folder (mass+calc)
        procPathMC = path + '/data/processed/mass+calc' + extraPath[i]
        # procPathCalc = 'data/processed/mass+calc' + extraPath[i]
       
        # Create new directories if needed
        if not os.path.exists(procPathMasses):
            os.makedirs(procPathMasses)
        
        if not os.path.exists(procPathCalc):
            os.makedirs(procPathCalc)
        
        if not os.path.exists(procPathMC):
            os.makedirs(procPathMC)
        print(rawPath)
        for filename in tqdm.tqdm(glob.glob(os.path.join(to_raw(rawPath), '*.txt'))):
            file = open (filename)
            lesion_type = file.readline()
            
            if lesion_type.find("MASS") != -1:
                if lesion_type.find("CALCIFICATION") == -1:
                    procPath = procPathMasses
                    countMasses += 1
                    count = countMasses
                else:
                    procPath = procPathMC
                    countMC += 1
                    count = countMC
            else:
                procPath = procPathCalc
                countCalc += 1
                count = countCalc
            
            
            # Calls functions from other scripts to encode the information in the '.txt' files
            attributes = encode_attributes(givenData=filename)
            label = get_label(givenData=filename)
    
          
            # Opens the correspondent image and creates new images with its patches in the floder with processed data
            imgName = filename[0:len(filename)-6] + '.png'
            maskName = filename[0:len(filename)-3] + 'npy'
            patch_arr = get_mask_patch(imgName, maskName, IMG_SIZE)
            patch = Image.fromarray(patch_arr)
            patch.save (procPath + '\\' + str(count) + '.png')
    
    
            # Adds information about the current file to the dictionaries
            if lesion_type.find("MASS") != -1:
                if lesion_type.find("CALCIFICATION") == -1:
                    yM["imgName"].append(str(count) + '.png')   
                    attrM["imgName"].append (str(count) + '.png')
                    attrM["label"].append (label)
                    attrM["attr1"].append(attributes[0])
                    attrM["attr2"].append(attributes[1])
                    attrM["subtlety"].append(attributes[2])  
                else:
                    yMC["imgName"].append(str(count) + '.png')   
                    attrMC["imgName"].append (str(count) + '.png')
                    attrMC["label"].append (label)
                    attrMC["attr1"].append(attributes[0])
                    attrMC["attr2"].append(attributes[1])
                    attrMC["attr3"].append(attributes[2])
                    attrMC["attr4"].append(attributes[3])
                    attrMC["subtlety"].append(attributes[4])
            
            else:
                yC["imgName"].append(str(count) + '.png')
                attrC["imgName"].append (str(count) + '.png')
                attrC["label"].append (label)
                attrC["attr1"].append(attributes[0])
                attrC["attr2"].append(attributes[1])
                attrC["subtlety"].append(attributes[2])
    
    
        # Uses Pandas library to convert the dictionaries information to .csv files
        yM_df = pd.DataFrame(yM)
        yM_df.to_csv(procPathMasses + '\\' + 'y.csv', header=True, index=False)
        attrM_df = pd.DataFrame(attrM)
        attrM_df.to_csv(procPathMasses + '\\' + 'attr.csv', header=True, index=False)
        
        yC_df = pd.DataFrame(yC)
        yC_df.to_csv(procPathCalc + '\\' + 'y.csv', header=True, index=False)
        attrC_df = pd.DataFrame(attrC)
        attrC_df.to_csv(procPathCalc + '\\' + 'attr.csv', header=True, index=False)
    
        yMC_df = pd.DataFrame(yMC)
        yMC_df.to_csv(procPathMC + '\\' + 'y.csv', header=True, index=False)
        attrMC_df = pd.DataFrame(attrMC)
        attrMC_df.to_csv(procPathMC + '\\' + 'attr.csv', header=True, index=False)

    print("Finished.")

if __name__ == '__main__':
    create_dataset()