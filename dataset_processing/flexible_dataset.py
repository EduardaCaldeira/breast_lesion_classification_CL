# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 11:14:48 2022

@author: Maria
"""

# Imports
import os
import sys
import random
from PIL import Image
import numpy as np
import pandas as pd
import checksumdir

# Project Imports
from encode_attributes import encode_attributes 
from create_dataset import create_dataset
from get_label import get_label



# Function: Check if data exists and if its contents are correct
def preprocess_if_needed(path, checksum):
    # check if data exists and if its contents are correct
    if os.path.isdir(path):
        string = checksumdir.dirhash(path)
        
        if string == checksum:
            return True
        
        return False



# Function: Get data set requirements
def get_requirements (requirements:str):
    
    lesionPath = []
   
    if requirements.upper().find('MASS') != -1:
        lesionPath.append('/masses')
    if requirements.upper().find('CALCIFICATION') != -1:
        lesionPath.append('/calcifications')
    if requirements.upper().find('MC') != -1:
        lesionPath.append('/mass+calc')
    if lesionPath == []:
        lesionPath = ['/masses', '/calcifications', '/mass+calc']
    
    label = get_label (requirements, False)
    
    extraPath = []
    if requirements.upper().find(' TRAIN') != -1:
        extraPath.append('/train')
    if requirements.upper().find(' VAL') != -1:
        extraPath.append('/val')
    if requirements.upper().find(' TEST') != -1:
        extraPath.append('/test')
    if extraPath == []:
        extraPath = ['/train', '/val', '/test']
        
    subtlety = 0
    for i in range (1, 6):
        if requirements.find(str(i)) != -1:
            subtlety = i
            break

    # if the user wants to search for mc files, we need to assume that each 
    # file can have up to 4 attributes, meaning it should be treated as an mc 
    # file
    for i in range (0, len(lesionPath)):
        if lesionPath[i] == '/mass+calc':
            requirements += ' MASS CALCIFICATION'
            break
    
    # the information regarding subtlety isn't necessary
    codedInfo = encode_attributes (requirements, False) 
    codedInfo = codedInfo [0:len(codedInfo)-1]
    
    return [lesionPath, label, codedInfo, subtlety, extraPath]



# Function: Get a specific database for a specific task and choose the split (train, validation, test)
def flexible_dataset (requirements:str, path:str = "C:/Users/Maria/Documents/GitHub/breast-cancer-classification/data/processed", checksum = '9e6d65e66d762f2473428987deb26ca4'):
    ''' 
    The input string should contain the specific features the user wants to
    search for, taking the following points into consideration:
        
        1. Search for (what you should write): masses (MASS), calcifications 
        (CALCIFICATION), mass+calcification (MC), validation (VAL), test (TEST),
        train (TRAIN), label (BENIGN or MALIGNANT), attribute (name of feature,
        OVAL, for example) and subtlety (a number between 1 and 5);
        2. The user can search cumulatively for more than one attribute (for 
        example, a shape and a margin; result -> all the images with that shape
        AND margin) but they CAN'T search for more than one value for the same 
        attribute (for example, two different shapes);
        3. The output obeys ALL the imposed criteria;
        4. If the user doesn't specify a certain feature, all the valid values 
        for that feature will be considered; 
        5. The different parts of the input should be separeted by a space and 
        the string should begin with a space (for example, ' MASS VAL OVAL SPICULATED 5'. 
    '''
        
    # TODO: Review and erase uppon testing
    # path = 'C:/Users/Maria/Documents/GitHub/breast-cancer-classification/data/processed'
    
    while not preprocess_if_needed(path=path, checksum=checksum):
        create_dataset()
    
    [lesionPath, label, codedInfo, subtlety, extraPath] = get_requirements(requirements)
    options = []
    fourAttributes = True
    lesionPathCode = [0, 0, 0]
    
    for i in range (0, len(lesionPath)):
        if lesionPath[i] == '/masses':
            lesionPathCode[0] = 1
        elif lesionPath[i] == '/calcifications':
            lesionPathCode[1] = 1
        else:
            lesionPathCode[2] = 1

    if label == 2:
        possibleLabels = ['0','1']
    else:
        possibleLabels = [str (label)]
    
    if subtlety == 0:
        possibleSub = list(range(0,6))
    else:
        possibleSub = [subtlety]
    
    if codedInfo[0] == '0':
        if lesionPathCode[1] == 1:
            possibleAttr1 = list(range(0,15))
        else:
            possibleAttr1 = list(range(0,10))
    else:
        possibleAttr1 = codedInfo[0]
    
    if codedInfo[1] == '0':
        possibleAttr2 = list(range(0,6))
    else:
        possibleAttr2 = codedInfo[1]
    
    if lesionPathCode[2] == 1:
        if codedInfo[2] == '0':
            possibleAttr3 = list(range(0,15))
        else:
            possibleAttr3 = codedInfo[2]
            
        if codedInfo[3] == '0':
            possibleAttr4 = list(range(0,6))
        else:
            possibleAttr4 = codedInfo[3]
    # if we aren't searching for files in 'mass+calc', encode_attributes will
    # only return 2 attributes => there is no need to append 2 extra ones
    else: 
        fourAttributes = False
        for lbl in range (0, len(possibleLabels)):
            for at1 in range (0, len(possibleAttr1)):
                for at2 in range (0, len(possibleAttr2)):
                    for sub in range (0, len(possibleSub)):
                        options.append([possibleLabels[lbl], str(possibleAttr1[at1]), str(possibleAttr2[at2]), str(possibleSub[sub])])
    
    images = []
    for lesion in range (0, len(lesionPath)):
        # checks if four attributes were considered and decides which of them 
        # are relevant for the current task
        if fourAttributes:
            options = []
            # only attributes 1 and 2 will be relevant
            if lesionPath[lesion] == '/masses':
                for lbl in range (0, len(possibleLabels)):
                    for at1 in range (0, len(possibleAttr1)):
                        for at2 in range (0, len(possibleAttr2)):
                            for sub in range (0, len(possibleSub)):
                                options.append([possibleLabels[lbl], str(possibleAttr1[at1]), str(possibleAttr2[at2]), str(possibleSub[sub])])
            # only attributes 3 and 4 will be relevant
            elif lesionPath[lesion] == '/calcifications':
                for lbl in range (0, len(possibleLabels)):
                    for at1 in range (0, len(possibleAttr3)):
                        for at2 in range (0, len(possibleAttr4)):
                            for sub in range (0, len(possibleSub)):
                                options.append([possibleLabels[lbl], str(possibleAttr3[at1]), str(possibleAttr4[at2]), str(possibleSub[sub])])
            # all attributes are important
            else:
                for lbl in range (0, len(possibleLabels)):
                    for at1 in range (0, len(possibleAttr1)):
                        for at2 in range (0, len(possibleAttr2)):
                            for at3 in range (0, len(possibleAttr3)):
                                for at4 in range (0, len(possibleAttr4)):
                                    for sub in range (0, len(possibleSub)):
                                        options.append([possibleLabels[lbl], str(possibleAttr1[at1]), str(possibleAttr2[at2]), str(possibleAttr3[at3]), str(possibleAttr4[at4]), str(possibleSub[sub])])

        for dataset in range (0, len(extraPath)):
            # TODO: Review and erase uppon testing
            filename = path + lesionPath[lesion] + extraPath[dataset] + '//attr.csv'
            # filename = os.path.join(path, lesionPath[lesion], extraPath[dataset], 'attr.csv')
            data = np.genfromtxt(filename, dtype='str', delimiter='\n')
            
            for line in range (0, len(data)):
                splittedInfo = data[line].split(',')
                
                # checks if the information extracted form the .csv file line 
                # is equal to one of the available options
                for opt in range (0, len(options)):
                    isValid = True
                    for feature in range (1, len(splittedInfo)):
                        if splittedInfo[feature].find(options[opt][feature - 1]) == -1:
                            isValid = False
                            break
                    # saves the names of the files with relevant information in
                    # variable 'images'
                    if isValid: 
                        images.append(filename[0:len(filename)-8] + splittedInfo[0])
                        break
                                
    return images



# Run to generate the database locally
if __name__ == "__main__":

    im = flexible_dataset('MASS TRAIN BENIGN')
    label = '0'

    attr = {'imgName': [],
            'label': []}

    path = 'C:/Users/Maria/Documents/GitHub/breast-cancer-classification'
    if path not in sys.path:
            sys.path.append(path)
    expPath = path + '/data/exp/masses/train'

    if not os.path.exists(expPath):
                os.makedirs(expPath)
                
    for index in range (0, len(im)):
        image = Image.open(im[index])
        image.save(expPath + '//' + str(index + 1) + '.png')
        attr["imgName"].append (str(index + 1) + '.png')
        attr["label"].append (label)

    defaultIndex = len(im)
    im = flexible_dataset('MASS TRAIN MALIGNANT')
    label = '1'

    for index in range (defaultIndex, len(im) + defaultIndex):
        image = Image.open(im[index - defaultIndex])
        image.save(expPath + '//' + str(index + 1) + '.png')
        attr["imgName"].append (str(index + 1) + '.png')
        attr["label"].append (label)

    attr_df = pd.DataFrame(attr)
    attr_df.to_csv(expPath + '//' + 'labels.csv', header=True, index=False)