import cv2
import numpy as np

from core.python_utils import rescale_array
from core.debugging import imshow

def resize(pixels, target_w, target_h, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(pixels, (target_w, target_h), interpolation=interpolation)

def scale(pixels, scale_factor, interpolation=cv2.INTER_LINEAR):
    target_w = int(pixels.shape[1]*scale_factor)
    target_h = int(pixels.shape[0]*scale_factor)
    return resize(pixels, target_w, target_h, interpolation)

def unique_quantize(plot, K, ignore_zero=False, eps=0.01, max_iter=100,max_tries=10):
    
    values = np.unique(plot)
    if ignore_zero:
        values = values[values > 0]
    
    Z = np.array([[value] for value in values],np.float32)
    
    ret, label, center = cv2.kmeans(Z,K,None,(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps
    ),max_tries,cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.ravel(center)
    
    errors = np.array([
        np.abs(plot-np.full_like(plot,center[i])) for i in range(len(center))
        ],float)

    classification = np.argmin(errors, axis=0)

    sortorder = np.argsort(center) # e.g. [1,2,0] means 1 maps to 0, 2 maps to 1, and 0 maps to 2
    inverse_sortorder = np.array([list(sortorder).index(i) for i in range(len(sortorder))],int) # finds the "to" index for a given "from" index
    
    center = center[sortorder]
    label = inverse_sortorder[classification]
    qplot = center[label]
    
    return center, label, qplot
    
def adaptive_threshold_by_mean(plot, blocksize):
    plot_image = np.expand_dims(np.clip(rescale_array(plot)*255,0,255).astype(np.uint8),axis=2)
    result=cv2.adaptiveThreshold(plot_image,1,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blocksize,0.0).squeeze() 
    return result > 0

def circular_structuring_element(R,dtype,remove_center=False):
    D = R*2+1
    def sieve(rMatrix,cMatrix):
        return np.logical_and(np.power(rMatrix-R,2) + np.power(cMatrix-R,2) <= R*R,
                                 np.logical_or(
                                     rMatrix!=R, cMatrix!=R
                                 ) if remove_center else np.ones((2*R+1,2*R+1),bool))
        
    return np.fromfunction(sieve,shape=(2*R+1,2*R+1),dtype=int).astype(dtype)
    
    
    