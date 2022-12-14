import sys
import cv2
import numpy as np
import scipy.ndimage
import os

import image_processing as IP
import utils as U

def get_rpe_nucleus_mask(image):

    image_LAB = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)

    L= U.rescale_array(image_LAB[:,:,0].astype(float))
    A= U.rescale_array(image_LAB[:,:,1].astype(float))
    B= U.rescale_array(image_LAB[:,:,2].astype(float))

    channel = 1.0-L

    centers, labels2D, qimg = IP.unique_quantize(channel,2,True)

    conservative_cell_mask = labels2D == 1

    adaptive_cell_mask = IP.Channel.adaptive_threshold_by_mean(channel,21)

    conservative_cell_mask = np.logical_and(conservative_cell_mask,adaptive_cell_mask)

    element = IP.circular_structuring_element(1,int,remove_center=True)
    
    ncounts = scipy.ndimage.convolve(conservative_cell_mask.astype(int),element)

    conservative_cell_mask[ncounts<2] = False

    return conservative_cell_mask