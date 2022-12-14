import cv2
import scipy.ndimage
import numpy as np

from core.python_utils import rescale_array
from core.image_ops import unique_quantize, adaptive_threshold_by_mean, circular_structuring_element

def get_conservative_cell_mask(pixels):
    
    pixels_LAB = cv2.cvtColor(pixels,cv2.COLOR_RGB2LAB)
    
    L, A, B = [rescale_array(pixels_LAB[:,:,c].astype(float)) for c in range(3)]
    
    channel = 1.0-L
    
    center, label, qplot = unique_quantize(channel, 2, True)
    
    conservative_cell_mask = label == 1
    
    adaptive_cell_mask = adaptive_threshold_by_mean(channel, 21)
    
    conservative_cell_mask = np.logical_and(conservative_cell_mask, adaptive_cell_mask)
    
    element = circular_structuring_element(1,int,remove_center=True)
    
    neighbor_counts = scipy.ndimage.convolve(conservative_cell_mask.astype(int), element)
    
    conservative_cell_mask[neighbor_counts < 2]= False
    
    return conservative_cell_mask
    

    
    