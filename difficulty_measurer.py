# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:10:22 2022

@author: Maria
"""


# Imports
import os
import sys
import tqdm
from PIL import Image
import pandas as pd


# TODO: Check this minor change (it is more flexible)
# appending the path to 'flexible_dataset' to the system
sys.path.append('C:/Users/Maria/Documents/GitHub/breast-cancer-classification/datasets')

from flexible_dataset import flexible_dataset

'''
# Append current working directory to PATH to export stuff outside this folder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


# Project Imports
from datasets.flexible_dataset import flexible_dataset
'''

def difficulty_measurer (extras:str):
    '''
    The main purpose of this function is the creation of a list with the lesions
    with the features specified in 'extras'; these lists allow us to order the
    lesions from easiest to harder; the ordering criterion is the subtlety level 
    of each lesion (1: subtle; 5: obvious)
    '''
    
    orderedDataset = []
    
    # extracts the directories of all the images from easiest to harder
    for subtlety in range (5, 0, -1):
        orderedDataset.append(flexible_dataset(extras + str(subtlety)))
        
    return orderedDataset
        
if __name__ == '__main__':

    path = 'C:/Users/Maria/Documents/GitHub/breast-cancer-classification'
    if path not in sys.path:
        sys.path.append(path)
      
    # I don't think test will be similar to this one -> TODO: simplify        
    extraPath = ['/test']
    labels = [' BENIGN ', ' MALIGNANT ']
    default = 0
    
    for i in tqdm.tqdm(range(len(extraPath))):  
        for sub in range(5):
            attr = {'imgName': [],
                    'label': []}
            
            for j in range(len(labels)):
                im = difficulty_measurer(' ' + extraPath[i][1:].upper() + labels[j])  
                expPath = path + '/data/cl/all' + extraPath[i] + 'Sub' + '/sub' + str(5 - sub)
                
                if not os.path.exists(expPath):
                    os.makedirs(expPath)
                    
                for index in range (0, len(im[sub])):                        
                    image = Image.open(im[sub][index])
                    image.save(expPath + '//' + str(index + 1 + default) + '.png')
                    attr["imgName"].append (str(index + 1 + default) + '.png')
                    attr["label"].append(j)
                    
                if j == 0:
                     default = len(im[sub])
                else:
                     default = 0

            attr_df = pd.DataFrame(attr)
            attr_df.to_csv(expPath + '//' + 'labels.csv', header=True, index=False)
    
    '''
    # TODO -> pôr isto de forma mais elegante        
    im = flexible_dataset(' VAL BENIGN')
    label = '0'

    attr = {'imgName': [],
            'label': []}

    expPath = path + '/data/cl/all/val'

    if not os.path.exists(expPath):
                os.makedirs(expPath)
                
    for index in range (0, len(im)):
        image = Image.open(im[index])
        image.save(expPath + '//' + str(index + 1) + '.png')
        attr["imgName"].append (str(index + 1) + '.png')
        attr["label"].append (label)
    

    defaultIndex = len(im)
    im = flexible_dataset(' VAL MALIGNANT')
    label = '1'

    for index in range (defaultIndex, len(im) + defaultIndex):
        image = Image.open(im[index - defaultIndex])
        image.save(expPath + '//' + str(index + 1) + '.png')
        attr["imgName"].append (str(index + 1) + '.png')
        attr["label"].append (label)

    attr_df = pd.DataFrame(attr)
    attr_df.to_csv(expPath + '//' + 'labels.csv', header=True, index=False)

    print("Finished.")
    '''
