# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 09:18:50 2022

@author: erica e manuel
"""



# Imports
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



# Custom Functions
def show_grid(patch, line):
    for i in range (0,5):
        plt.subplot(1,5,i+1)  #not sure
        img=Image.open(patch[line][i])
        img1=np.array(img)
        plt.imshow(img1, cmap="gray")
        img.close()
    
    plt.show()

    return


# Test
if __name__ == "__main__":
    
    # Define paths
    path = r'C:\Users\efgom\OneDrive\Documentos\GitHub\breast-cancer-classification\train'

    choice = input('Select mass (M), calcification(C) or both(B): ')

    patch = (
        [["", "", "", "", ""],
        ["", "", "", "", ""]]
        )


    for filename in glob.glob(os.path.join(path, '*.txt')):
        file = open(filename)
        lesion_type = file.readline()
        pathology = file.readline()
        assessment = file.readline()
        subtlety = int(file.readline())
        
        if(choice=='B'):
            if(lesion_type[0]=='M'):
                patch[0][subtlety-1]=filename.replace("_0.txt", ".png")
            else:
                patch[1][subtlety-1]=filename.replace("_0.txt", ".png")
    
        else:
            if(lesion_type[0]==choice):
                patch[0][subtlety-1]=filename.replace("_0.txt", ".png")
            

                
                
                
    if (choice=='B'):
        # Mass
        plt.figure(1)  
        show_grid(patch, 0)
        
        # Calcification
        plt.figure(2)         
        show_grid(patch, 1)


    else:
        show_grid(patch, 0)
