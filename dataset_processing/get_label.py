# Imports
import numpy as np



# Function: Auxiliary function that returns the label given a text file (0 for benign and 1 for malignant).
def get_label(givenData: str, isFile:bool = True):
    
    if isFile:
        # Load data from text file
        data = np.genfromtxt(givenData, dtype='str', delimiter='\n')
    
        # Get the pathology
        pathology = data[1]


        # Check if data is not corrupted
        assert pathology.upper().startswith('BENIGN') or pathology.upper() == 'MALIGNANT', 'Error on the text file description, regarding the pathology'

    # as used in flexible_dataset.py
    else: 
        pathology = givenData.upper()
        if pathology.find('BENIGN') == -1 and pathology.find('MALIGNANT') == -1:
            return 2

    # Create a label variable
    label = -1


    # Assign positive label
    if pathology.upper().find('BENIGN') != -1:
        label = 0


    # Assign negative label
    else:
        label = 1
    

    return label



# Test
if __name__ == "__main__":
    
    # Define data path
    PATH = 'data/ddsm/train/'
    FILE = 'benign_01_3091_LEFT_CC_0'
    FILENAME = PATH + FILE + '.txt'

    # Expected result: [4 2 5]
    print(get_label(filename=FILENAME))
