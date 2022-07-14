# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:06:31 2022

@author: Maria
"""

import numpy as np
from PIL import Image

def get_patch(img, coords_tuple, line_len, col_len = 0):
    if col_len == 0:
        col_len = line_len
        
    patch = np.zeros((line_len, col_len))
    for line in range (coords_tuple[0] - (int) (line_len/2), coords_tuple[0] + (int) (line_len/2) + line_len % 2):
        for col in range (coords_tuple[1] - (int) (col_len/2), coords_tuple[1] + (int) (col_len/2) + col_len % 2):
            patch[line - coords_tuple[0] + (int) (line_len/2)][col - coords_tuple[1] + (int) (col_len/2)] = img[line][col]

    return patch

def show_patch (mask, isSquare):
    for i in range (0, mask.shape[0]):
        if mask[i].sum() != 0:
            stLine = i
            break
    for j in range (mask.shape[0] - 1, -1, -1):
         if mask[j].sum() != 0:
            lastLine = j
            break
    for k in range (0, mask.shape[1]):
        if mask[:,k].sum() != 0:
            stCol = k
            break
    for l in range (mask.shape[1] - 1, -1, -1):
        if mask[:,l].sum() != 0:
            lastCol = l
            break 
    
    col_center = (int) ((lastCol + stCol + 1)/2)
    lin_center = (int) ((lastLine + stLine + 1)/2)
    numCols = lastCol - stCol + 1
    numLines = lastLine - stLine + 1
 
    # scale conversion (0-1 to 0-255) to allow the user to interpret the resultant image
    if isSquare:
        myPatch = get_patch (mask, (lin_center, col_center), max (numLines, numCols)) * 255
    else:
        myPatch = get_patch (mask, (lin_center, col_center), numLines, numCols) * 255
        
    img = Image.fromarray(myPatch)  
    img.show()
    
    return (myPatch)