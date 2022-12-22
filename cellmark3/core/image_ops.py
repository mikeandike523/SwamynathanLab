from math import floor

import cv2
import numpy as np
from scipy.spatial import KDTree
import skimage.draw
import shapely.geometry

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
    def sieve(rMatrix,cMatrix):
        return np.logical_and(np.power(rMatrix-R,2) + np.power(cMatrix-R,2) <= R*R,
                                 np.logical_or(
                                     rMatrix!=R, cMatrix!=R
                                 ) if remove_center else np.ones((2*R+1,2*R+1),bool))
        
    return np.fromfunction(sieve,shape=(2*R+1,2*R+1),dtype=int).astype(dtype)

def draw_element_on(image, x, y, element, color):
    
    x = int(floor(x))
    y = int(floor(y))
    
    eH, eW = element.shape
    
    offsx = int(eW//2)
    offsy = int(eH//2)
    
    new_image = image.copy()
    
    for r in range(eH):
        for c in range(eW):
            try:
                if element[r,c]:
                    new_image[y-offsy+r, x-offsx+c,...] = color
            except IndexError as e:
                pass
            
    return new_image

def connected_components(mask, connectivity=4):
    
    mask_image = np.expand_dims(np.where(mask,255,0).astype(np.uint8),axis=2)
    
    labels = cv2.connectedComponents(mask_image, connectivity=connectivity)[1].squeeze()
    
    labels[labels==0] = -1
    
    labels[labels>=1] -= 1
    
    return labels
            
def voronoi(hint):
    
    print("Running voronoi...")
    
    seed_pixel_locations_rc = np.transpose(np.nonzero(hint > 0))
    
    seed_pixel_locations_rc_tree = KDTree(seed_pixel_locations_rc)
    
    H, W = hint.shape
    
    zero_locations_rc = np.transpose(np.nonzero(hint == 0))
    
    markers = hint.copy()
    
    cluster_map = connected_components(hint!=-1,4)
    
    for r,c in zero_locations_rc:
        d, i = seed_pixel_locations_rc_tree.query((r,c),1)
        nr, nc = seed_pixel_locations_rc[i]
        if cluster_map[nr,nc]!=cluster_map[r,c]:
            markers[r,c] = -1
        else:
            markers[r,c] = hint[nr,nc]
    
    print("Done.")  
        
    return markers

def distance_transform_L1(mask):
    
    mask_image = np.expand_dims(np.where(mask,255,0).astype(np.uint8),axis=2)
    
    dist = cv2.distanceTransform(mask_image, cv2.DIST_L1, 3)
    
    return dist.squeeze().astype(int)

def binarize_otsu(plot):
    
    plot_image = np.expand_dims(255*plot,axis=2).astype(np.uint8)
    
    binarized = cv2.threshold(plot_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    return binarized.squeeze() > 0

def rasterize_polygon(points_rc, shape = None):
    
    points_array = np.array(points_rc,int)
    
    if shape is None:
        
        min_r = np.min(points_array[:,0])
        min_c = np.min(points_array[:,1])

        max_r = np.max(points_array[:,0])
        max_c = np.max(points_array[:,1])

        shape = max_r - min_r + 1, max_c - min_c + 1
    
    return skimage.draw.polygon2mask(shape,points_array)

def get_outer_contour(mask): # contours are in x, y
    # cases to handle: where there is more than on outer contour
    # where there are no outer contours
    
    mask_image = np.expand_dims(np.where(mask,255,0),axis=2).astype(np.uint8)
    contours, heirarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = [contour.squeeze(axis=-2) for contour in contours]
    heirarchy = heirarchy.squeeze(axis=0)
    
    outer_contours = []
    
    for contour, heirarchy in zip(contours,heirarchy):
        if heirarchy[-1] == -1:
            outer_contours.append(contour)
            
    if len(outer_contours)==0  or len(outer_contours) > 1:
        return None
    
    return outer_contours[0]

def check_if_contour_bounds_pixels_in_mask(contour, mask, minimum_coverage_fraction = 1.0, dilate_radius = 2):
    
    if len(contour) < 3: return False
    
    if np.count_nonzero(mask) == 0: return False
    
    rc_contour = np.flip(np.copy(contour), axis=1)
    
    rc_contour_mask = rasterize_polygon(rc_contour, mask.shape)
    
    dilation_element = np.expand_dims(circular_structuring_element(dilate_radius,np.uint8),axis=2)
    
    if dilate_radius > 0:
        rc_contour_mask =cv2.dilate(np.expand_dims(np.where(rc_contour_mask,1,0),axis=2).astype(np.uint8), dilation_element).squeeze() > 0 
    
    intersection = np.logical_and(mask, rc_contour_mask)
    
    return np.count_nonzero(intersection) / np.count_nonzero(mask) >= minimum_coverage_fraction

def resample_contour_by_segment_length(contour, segment_length_pixels, closed=False):
    
    # Does not convert to nearest integer. That will be the job of the calling code.
    # However, the result contour will inherit the dtype of the input contour if present
    
    # Will always round such that the largest segment is still <= segment_length_pixels
    
    resampled_contour = []
    
    poly = None
    
    if closed:
        poly = shapely.geometry.LinearRing(contour)
    else:
        poly = shapely.geometry.LineString(contour)
    
    num_segments = int(np.ceil(poly.length / segment_length_pixels))
    
    for it in range((num_segments+1) if not closed else num_segments):
        
        t = it/num_segments
        
        interpolated_point = poly.interpolate(t,normalized=True)
        
        resampled_contour.append(np.array(interpolated_point.coords[0]))
    
    return np.array(resampled_contour, contour.dtype if hasattr(contour, 'dtype') else float)

    
    
    