# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:17:13 2022

@author: Maria e Manuel
"""
import numpy as np
import cv2

def get_patch(img, coords_tuple, size):
    import warnings
    warnings.warn("Deprecated, use datasets.auxiliary.get_patch instead")
    x,y = coords_tuple
    half_size = size//2
    # pad image so that we can get patches from edges
    img = np.pad(img, half_size)
    
    # the patch is now obtained by looking at the coordinates
    # (x:x+half_size, y:y + half_size), since we padded the image
    patch = img[x:x+half_size, y:y+half_size]
    return patch

if __name__=="__main__":
    #test
    img = np.array ([[1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25]])
    
    patch_even = get_patch(img, (2,2), 2)
    patch_odd = get_patch(img, (2,2), 3)
    print (patch_even)
    print (patch_odd)