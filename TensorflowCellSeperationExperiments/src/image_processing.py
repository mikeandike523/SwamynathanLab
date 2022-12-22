from dataclasses import asdict
import numpy as np
from numba import float32, uint32
from numba import jit
from numba.experimental import jitclass
import PIL
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import scipy.ndimage
import copy
import sklearn.cluster
import math
import shapely.geometry
import scipy.interpolate
import rasterio.features
import alphashape
import warnings
from types import SimpleNamespace as SNs

import utils as U #image_processing.py will certainly be dependent on utils.py
                    # utils.py cannot be dependent on any other project file

def imload(path):
    return np.asarray(PIL.Image.open(path))

def imsave(pixels, path):
    PIL.Image.fromarray(pixels).save(path)

def image_1p0_to_255(img):
    return (255*img).astype(np.uint8)

def rescale_array(arr):
    if np.count_nonzero(arr) == 0:
        return np.zeros_like(arr)
    if np.all(arr==np.max(arr)):
        return np.ones_like(arr)
    return (arr.copy()-np.min(arr))/(np.max(arr) - np.min(arr))

def generic_array_to_RGB_image(arr):
    arr_copy = np.squeeze(arr.copy())
    if arr_copy.ndim == 2: # array is float valued and in range 0.0-1.0
        arr_copy = greyscale_plot_to_color_image(arr_copy)
    #else the array is assumed to already be an image
    return arr_copy

def island_to_mask(island,W=None,H=None):
    island_arr = np.array(island,float).copy()
    bounds = np.min(island_arr[:,0]),np.min(island_arr[:,1]),np.max(island_arr[:,0]),np.max(island_arr[:,1]),
    mask_W = int(bounds[2] - bounds[0] + 1)
    mask_H = int(bounds[3] - bounds[1] + 1)
    top_left_corner = np.array((bounds[0],bounds[1]),float)
    if W is not None:
        mask_W = int(W)
        mask_H = int(H)
        top_left_corner = np.array((0,0),float)
    mask = np.zeros((mask_H,mask_W),bool)
    for point in island_arr:
        x, y = (point-top_left_corner).astype(int)
        mask[y,x] = True
    return mask

#points: (x,y) coordinates
def run_alphashape_and_get_polygons(points, alpha):

    shape = alphashape.alphashape(points,alpha)
    
    # According to https://alphashape.readthedocs.io/en/latest/alphashape.html#:~:text=Returns%3A,the%20resulting%20geometry
    # Shape can be of three relevant types:

    # shapely.geometry.Polygon
    # shapely.geometry.Point
    # shapely.geometry.LineString

    # but can it be the multi variant of these?
    # Assume so

    polygons = []

    def recursive_find_polygons(geom):
        nonlocal polygons

        if isinstance(geom,shapely.geometry.Point) or isinstance(geom,shapely.geometry.MultiPoint):
            warnings.warn(f"Found alpha shape geom of type {type(geom)}.")

        if isinstance(geom, shapely.geometry.LineString):
            try:
                poly = shapely.geometry.Polygon(geom.coords)
            except Exception as e:
                print(e)
                warnings.warn('Found alpha shape geom of type LineString that could not be converted to type Polygon.')

        if isinstance(geom, shapely.geometry.MultiLineString):
            for geom in geom.geoms:
                recursive_find_polygons(geom)

        if isinstance(geom, shapely.geometry.Polygon):
            polygons.append(geom)

        if isinstance(geom, shapely.geometry.MultiPolygon):
            for geom in geom.geoms:
                recursive_find_polygons(geom)

    recursive_find_polygons(shape)

    return polygons

def bounds_of_island(island):
    tlc = np.array([np.min(island[:,0]),np.min(island[:,1])],island.dtype)
    brc = np.array([np.max(island[:,0]),np.max(island[:,1])],island.dtype)
    return tlc, brc

def corners_to_slice(tlc,brc):
    return np.s_[tlc[1]:(brc[1]+1),tlc[0]:(brc[0]+1)]

# def fast_partion_mask_along_curve(mask,curve,start_index,end_index,dividerLineThickness =3, dividerLineLength = 250, spillover_pixels = 3):

#     tangents, normals = Curve.curve_tangents_and_normals(curve)

#     mask_image = np.dstack((np.where(mask,255,0),)*3).astype(np.uint8)

#     for idx in [start_index,end_index]:
#         origin=np.array(curve[idx],float)
#         normal = np.array(normals[idx],float)
#         A = (origin+dividerLineLength/2*t=normal).astype(int)
#         B = (origin+dividerLineLength/2*normal).astype(int)
#         mask_image = cv2.line(mask_image,A,B,(0,0,0),dividerLineThickness)

#     divided_mask = mask_image[:,:,0] > 0

#     grouping = BooleanImageOps.connected_components(divided_mask,4)

#     islandsDict = Grouping.getIslands(grouping)

#     maintained_islands = []

#     for label, island in islandsDict.items():

#         island_mask = island_to_mask(island,mask.shape[1],mask.shape[0])

#         island_mask_dilated


    
class Pallete:

    # --- Courtesy of https://coolors.co/palettes/popular/contrast
    LEGEND_COLORS = [{"name":"Atomic Tangerine","hex":"ef9c70","rgb":[239,156,112],"cmyk":[0,35,53,6],"hsb":[21,53,94],"hsl":[21,80,69],"lab":[72,26,36]},{"name":"Brown Sugar","hex":"b3653d","rgb":[179,101,61],"cmyk":[0,44,66,30],"hsb":[20,66,70],"hsl":[20,49,47],"lab":[51,28,36]},{"name":"Dark Liver","hex":"44414c","rgb":[68,65,76],"cmyk":[11,14,0,70],"hsb":[256,14,30],"hsl":[256,8,28],"lab":[28,4,-6]},{"name":"Coyote Brown","hex":"6f5838","rgb":[111,88,56],"cmyk":[0,21,50,56],"hsb":[35,50,44],"hsl":[35,33,33],"lab":[39,5,22]},{"name":"Gold Fusion","hex":"867959","rgb":[134,121,89],"cmyk":[0,10,34,47],"hsb":[43,34,53],"hsl":[43,20,44],"lab":[51,0,19]},{"name":"Ebony","hex":"666b55","rgb":[102,107,85],"cmyk":[5,0,21,58],"hsb":[74,21,42],"hsl":[74,11,38],"lab":[44,-6,12]},{"name":"Gray X 11 Gray","hex":"b8b9be","rgb":[184,185,190],"cmyk":[3,3,0,25],"hsb":[230,3,75],"hsl":[230,4,73],"lab":[75,1,-3]},{"name":"Roman Silver","hex":"797f8d","rgb":[121,127,141],"cmyk":[14,10,0,45],"hsb":[222,14,55],"hsl":[222,8,51],"lab":[53,1,-8]},{"name":"Almond","hex":"eadac3","rgb":[234,218,195],"cmyk":[0,7,17,8],"hsb":[35,17,92],"hsl":[35,48,84],"lab":[88,2,13]},{"name":"Black Chocolate","hex":"1d1812","rgb":[29,24,18],"cmyk":[0,17,38,89],"hsb":[33,38,11],"hsl":[33,23,9],"lab":[9,1,5]}]
    # ---

    @classmethod
    def legend_color(cls,i):
        return Pallete.LEGEND_COLORS[i%len(Pallete.LEGEND_COLORS)]["rgb"]
    
    def __init__(self):
        self.idx = 0

    def nextColor(self):
        last_idx = self.idx
        self.idx = (self.idx + 1) % len(Pallete.LEGEND_COLORS)
        return Pallete.LEGEND_COLORS[last_idx]["rgb"]

channel_numba_spec = [
    ('W',uint32),
    ('H',uint32),
    ('data',float32[:,:])
]

@jitclass(channel_numba_spec)
class nbChannel:
    
    def __init__(self, W, H):
        self.W = W
        self.H = H
        self.data = np.zeros((H,W),np.float32)
       
    def point_is_in_bounds(self,x,y):
        return x >=0 and x < self.W and y >= 0 and y < self.H
    
    def read(self,x,y):
        if self.point_is_in_bounds(x,y):
            return self.data[y,x]
        return 0
    def write(self,x,y,v):
        if self.point_is_in_bounds(x,y):
            self.data[y,x] = v

    def read_window(self,x,y,w,h):
        result = np.zeros((h,w),dtype=np.float32)
        for dx in range(w):
            for dy in range(h):
                xx = x + dx
                yy = y + dy
                if self.point_is_in_bounds(xx,yy):
                    result[dy,dx] = self.data[yy,xx]
        return result

    def width(self):
        return self.W

    def height(self):
        return self.H

    def rescaled(self):
        if np.count_nonzero(self.data) == 0:
            return np.zeros_like(self.data)
        if np.all(self.data==1):
            return np.ones_like(self.data)
        return (self.data.copy() - np.min(self.data))/(self.data - np.min(self.data))
    
def binarize_otsu(arr,rescale_first = True):
    input_arr = arr.copy()
    if rescale_first:
        input_arr = rescale_array(input_arr)
    gsimage = np.expand_dims((input_arr * 255).astype(np.uint8),axis=2)
    _, thresholded_image = cv2.threshold(gsimage,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholded_image = np.squeeze(thresholded_image)
    return thresholded_image[:,:] > 0

class Channel:
    

    @classmethod
    def Canny(cls,arr,low,high):
        canny_input = greyscale_plot_to_color_image(arr.copy())
        return np.squeeze(cv2.Canny(canny_input,low,high)) > 0

    @classmethod
    def from_array(cls,arr):
        H, W = arr.shape[:2]
        ch = nbChannel(W,H)
        for x in range(W):
            for y in range(H):
                ch.write(x,y,arr[y,x])
        return ch

    @classmethod
    def to_array(cls,ch):
        plot = np.zeros((ch.height(),ch.width()),dtype=float)
        for x in range(ch.width()):
            for y in range(ch.height()):
                plot[y,x] = ch.read(x,y)
        return plot

    @classmethod
    def to_greyscale_image(cls,nbch):
        img = np.zeros((nbch.height(),nbch.width(),3),dtype=np.uint8)
        for x in range(nbch.width()):
            for y in range(nbch.height()):
                grey_level = int(255*nbch.read(x,y))
                img[y,x,:] = (grey_level,grey_level,grey_level)
        return img
        

    @classmethod
    def rescaled_nbch(cls, ch):
        arr = Channel.to_array(ch)
        arr = rescale_array(arr)
        return Channel.from_array(arr)

    @classmethod
    def convolve_arrays(cls,image,kernel,border=0):
        element = kernel.astype(float)
        result = scipy.ndimage.convolve(image,element,mode='constant',cval=border)
        return result

    @classmethod
    def adaptive_threshold_by_mean(cls,arr,blocksize):
        arr_image = np.expand_dims((255*arr).astype(np.uint8),axis=2)
        return np.squeeze(cv2.adaptiveThreshold(arr_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blocksize,0.0)) == 255

    @classmethod
    def resize_as_image(cls,arr,W,H):
        
        W = int(W)
        
        H = int(H)
        
        arr_img = greyscale_plot_to_color_image(arr)
        
        arr_img = cv2.resize(arr_img,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
        
        return np.mean(arr_img.astype(float),axis=2)/255

def imshow(image,name="imshow"):
    PIL.Image.fromarray(image).save(f"debug/{name}.png")

def imshowjpg(image,name="imshow", quality=85):
    PIL.Image.fromarray(image).save(f"debug/{name}.jpg",quality=quality)

def scale_image_nearest(rgbImageU8, scale_factor_x_or_xy, scale_factor_y = None):
    return scale_image(cv2.INTER_NEAREST,rgbImageU8,scale_factor_x_or_xy, scale_factor_y)

def scale_image_linear(rgbImageU8, scale_factor_x_or_xy, scale_factor_y = None):
    return scale_image(cv2.INTER_LINEAR,rgbImageU8,scale_factor_x_or_xy, scale_factor_y)

def scale_image_cubic(rgbImageU8, scale_factor_x_or_xy, scale_factor_y = None):
    return scale_image(cv2.INTER_CUBIC,rgbImageU8,scale_factor_x_or_xy, scale_factor_y)

def scale_image(interp_mode, rgbImageU8, scale_factor_x_or_xy, scale_factor_y = None):
    scale_factor_x = scale_factor_x_or_xy
    if scale_factor_y is None:
        scale_factor_y = scale_factor_x_or_xy
    H, W = rgbImageU8.shape[:2]
    newH = int(scale_factor_y *H)
    newW = int(scale_factor_x *W)
    return cv2.resize(rgbImageU8,dsize=(newW,newH),interpolation = interp_mode)

def unique_quantize(arr, K,ignore_zero=False, eps = 0.05, max_iter = 100, max_tries = 20):

    """@param arr: 2D numpy array of floats"""

    H, W = arr.shape

    unique_values = np.squeeze(np.unique(arr.copy()))

    unique_values = np.array(unique_values, float)

    unique_values = unique_values[unique_values!=0]

    if unique_values.ndim == 0:
        unique_values = np.array([unique_values],float)

    unique_values = np.ravel(unique_values)

    unique_values = np.expand_dims(unique_values,1)

    Z = unique_values.astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,max_iter,eps)

    compactness, labels, centers = cv2.kmeans(Z,K,None,criteria,max_tries,cv2.KMEANS_RANDOM_CENTERS)

    labels = np.ravel(np.squeeze(labels))
    centers = np.ravel(np.squeeze(centers))

    sortorder = np.argsort(centers) # old index --> index to sortorder

    inverse_sortorder = np.array([list(sortorder).index(i) for i in range(len(centers))],int)

    ret_center = centers[sortorder]
    ret_labels2D = np.zeros((H,W),int)
    ret_qimg = np.zeros((H,W),float)

    errors = [np.power((arr-center),2) for center in centers]
    errors = np.array(errors,float)

    classification = np.squeeze(np.argmin(errors,axis=0))

    ret_labels2D = inverse_sortorder[classification]

    if ignore_zero:
        ret_labels2D[arr==0.0] = -1

    ret_qimg = centers[classification]

    if ignore_zero:
        ret_qimg[arr==0.0] = 0.0

    # for x in range(W):
    #     print("Assigned {:.2f}%".format(100 * (x+1)/W))
    #     for y in range(H):
    #         #center_id = np.argmin([(arr[y,x]-center)**2 for center in centers])
    #         center_id = np.argmin(errors[:,y,x])
    #         ret_labels2D[y,x] = sortorder.index(center_id)
    #         ret_qimg[y,x] = centers[center_id]

    # LUT = np.zeros((256,),int)
    # brightness_LUT = np.zeros((256,),float)
    # for i in range(256):
    #     amt = i/255
    #     center_id = np.argmin([(center-amt)**2 for center in centers])
    #     LUT[i] = center_id
    #     brightness_LUT[i] = amt
    # arr_256 = (255*arr).astype(np.uint8)
    # unsorted_labels2D = LUT[arr_256]
    
    # ret_labels2D = np.zeros_like(unsorted_labels2D)

    # for center_id in range(len(centers)):
    #     ret_labels2D[unsorted_labels2D == center_id] = sortorder.index(center_id) 

    # ret_qimg = brightness_LUT[arr_256]

    return np.array(ret_center,float), np.array(ret_labels2D,int), np.array(ret_qimg,float)

def dbscan_group_cityblock(boolean_array,max_cityblock_distance=1):
    H, W = boolean_array.shape
    feature_set = np.transpose(np.nonzero(boolean_array))
    scanner = DBSCAN(eps=max_cityblock_distance,metric='cityblock',min_samples=1)
    clustering_result = scanner.fit(feature_set)
    labels = np.ravel(np.squeeze(clustering_result.labels_))
    grouping = np.zeros((H,W),int) - 1
    for idx,feature in enumerate(feature_set):
        y, x = feature
        grouping[y,x] = labels[idx]
    return grouping


def dbscan_group_cityblock_in_patches(boolean_array,patch_size,max_cityblock_distance=1):
    
    H, W = boolean_array.shape
    
    w = patch_size
    h = patch_size
    
    locations = U.Geometry.get_window_locations_covering_image(W,H,patch_size,patch_size)
    
    mask = WindowableArray(boolean_array)
    
    result = np.zeros((H,W),int) - 1
    
    result = WindowableArray(result)
    
    def is_trivial(image_section):
        return np.all(np.mean(image_section)==image_section)
    
    for x, y in locations:
        
        mask_section = mask.get_window(x,y,w,h)
        
        if is_trivial(mask_section):
            continue
        
        else:
            
            grouping = dbscan_group_cityblock(mask_section,max_cityblock_distance)
        
            U.dprint(np.any(grouping!=-1))
        
            result.set_window(x,y,w,h,grouping)
    
    return result.get_window(0,0,W,H,-1)


def random_color():
    
    """Random colors do not cover entire color gambit.
    They are restricted to be readable as well as neither white nor black"""

    
    color = np.array([np.random.uniform(0.15,0.85),np.random.uniform(0.15,0.85),np.random.uniform(0.15,0.85)],float)
    return (255*color).astype(np.uint8)

used_random_colors = []
def random_color_at_index(i):    
    if i<len(used_random_colors):
        return used_random_colors[i]

    """Colors are not guaranteed unique"""
    while len(used_random_colors) < i+1:
        used_random_colors.append(random_color())
    return used_random_colors[i]

def delete_color_at_index(i):
    global used_random_colors
    used_random_colors.pop(i)

def get_color_cache():
    return used_random_colors

def set_color_cache(color_cache):
    global used_random_colors
    used_random_colors = color_cache
    

def grouping_to_color_image(grouping):
    # num_labels = np.max(grouping) + 1
    # H, W = grouping.shape
    # grouping_image = np.full((H,W,3),(255,255,255),dtype=np.uint8)
    # for x in range(W):
    #     for y in range(H):
    #         if grouping[y,x] != -1:
    #             grouping_image[y,x,:] = random_color_at_index(grouping[y,x])

    num_labels = np.max(grouping) + 1

    # Ensure the requisite number of labels has been generated at least once

    for i in range(num_labels):
        _ = random_color_at_index(i)

    grouping_image = np.full(tuple(grouping.shape)+(3,),255, dtype=np.uint8)

    for i in range(num_labels):
        color = random_color_at_index(i)
        grouping_image[grouping==i, :] = color

    return grouping_image

def greyscale_plot_to_color_image(plot):
    channel = (255*np.array(plot,dtype=float)).astype(np.uint8)
    return np.dstack((channel,)*3)

def circular_structuring_element(radius,dtype=np.uint8,**kwargs):
    R = int(radius)
    element = np.zeros((2*R+1,2*R+1),dtype=dtype)
    for dx in range(-R,R+1):
        for dy in range(-R,R+1):
            if dx**2 + dy **2 <= R**2:
                x = dx +R
                y = dy+ R
                element[y,x] = 1
    if "remove_center" in kwargs and kwargs["remove_center"]:
        element[R,R] = 0
    return element

class BooleanImageOps:

    @classmethod
    def select_image_region_by_mask(cls,image,mask,bgcolor):
        clipped_mask = BooleanImageOps.clipped_mask(mask)
        tlc, brc = BooleanImageOps.bounds(mask)
        mask_slice = corners_to_slice(tlc,brc)
        image_section = image[mask_slice].copy()
        image_section = BooleanImageOps.apply_mask_to_image(clipped_mask,image_section,bgcolor)
        return image_section
    
    @classmethod
    def boolean_to_color(cls,mask,color=(255,255,255),bgcolor=(0,0,0)):
        H, W = mask.shape
        image=np.full((H,W,3),bgcolor,dtype=np.uint8)
        image[mask,:] = color
        return image

    @classmethod
    def masks_to_image(cls,masks,bgcolor=(0,0,0)):
        """!    Visualize a sequence of masks with the default legend colors"""
        H, W = masks[0].shape
        image = np.zeros((H,W,3),dtype=np.uint8)
        for i,mask in enumerate(masks):
            image[mask] = cls.boolean_to_color(mask,Pallete.legend_color(i))
        return image

    @classmethod
    def draw_line_on(cls,img,A,B,T=1):
        H, W = img.shape
        lm=Draw.line_mask(A,B,W,H,T)
        new_img = img.copy()
        new_img = np.logical_or(new_img,lm)
        return new_img

    @classmethod
    def draw_curve_on(cls,mask,curve,T=1,closed=False):
        
        mask_copy = mask.copy()

        get_curve_point = lambda i: curve[i%len(curve)]

        for iA in range(len(curve) if closed else len(curve)-1):
            A = get_curve_point(iA)
            B = get_curve_point(iA+1)
            mask_copy = BooleanImageOps.draw_line_on(mask_copy,A,B,T)

        return mask_copy

    @classmethod
    def erodeWithKernel(cls,boolean_image,kernel_or_size,iterations=1):
        H, W = boolean_image.shape
        element = kernel_or_size
        if isinstance(kernel_or_size, int):
            element = circular_structuring_element(kernel_or_size) # default to using a circular structuring element
        boolean_image_rgb = np.zeros((H,W,3),np.uint8)
        boolean_image_rgb[boolean_image,...] = 255
        boolean_image_rgb = cv2.erode(boolean_image_rgb,element,iterations=iterations)
        eroded_boolean_image = boolean_image_rgb[:,:,0] > 0 #r, g, b channels should be identical after cv2.erode
        return eroded_boolean_image
        
    @classmethod
    def dilateWithKernel(cls,boolean_image,kernel_or_size,iterations=1):
        H, W = boolean_image.shape
        element = kernel_or_size
        if isinstance(kernel_or_size, int):
            element = circular_structuring_element(kernel_or_size) # default to using a circular structuring element
        boolean_image_rgb = np.zeros((H,W,3),np.uint8)
        boolean_image_rgb[boolean_image,...] = 255
        boolean_image_rgb = cv2.dilate(boolean_image_rgb,element,iterations=iterations)
        eroded_boolean_image = boolean_image_rgb[:,:,0] > 0 #r, g, b channels should be identical after cv2.erode
        return eroded_boolean_image

    @classmethod
    def openWithKernel(cls, boolean_image, kernel_or_size,iterations=1,repeats=1):
        boolean_image_copy = boolean_image.copy()
        for _ in range(repeats):
            boolean_image_copy = BooleanImageOps.dilateWithKernel(BooleanImageOps.erodeWithKernel(boolean_image_copy,kernel_or_size,iterations),kernel_or_size,iterations)
        return boolean_image_copy

    @classmethod
    def closeWithKernel(cls, boolean_image, kernel_or_size,iterations=1,repeats=1):
        boolean_image_copy = boolean_image.copy()
        for _ in range(repeats):
            boolean_image_copy = BooleanImageOps.erodeWithKernel(BooleanImageOps.dilateWithKernel(boolean_image_copy,kernel_or_size,iterations),kernel_or_size,iterations)
        return boolean_image_copy

    @classmethod
    def apply_mask_to_image(cls,mask,image,bgcolor=(0,0,0)):
        new_image = np.full_like(image,bgcolor)
        new_image[mask,...] = image[mask,...]
        return new_image

    @classmethod
    def connected_components(cls, mask, connectivity = 4):
        mask_image = np.expand_dims(np.where(mask,1,0).astype(np.uint8),axis=2)
        markers = np.squeeze(cv2.connectedComponents(mask_image,connectivity)[1].astype(int)-1)
        return markers

    @classmethod
    def get_longest_contour(cls,mask):
        """
        Length is measured in euclidean distance.
        This can be used to get the border of mask representing a single island of pixels.
        """

        mask_image = np.expand_dims(np.where(mask,255,0),axis=2).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour.squeeze() for contour in contours]
        return contours[np.argmax([Curve.measure_length(contour) for contour in contours])]

    @classmethod
    def distance_transform_L1(cls,mask):
        mask_image = np.expand_dims(np.where(mask,255,0).astype(np.uint8),axis=2)
        dst = cv2.distanceTransform(mask_image,cv2.DIST_L1,3)
        dst = np.squeeze(dst)
        return dst

    @classmethod
    def fast_filter_small_islands_and_get_grouping(cls, mask, connectivity, min_pixels):

        result = np.zeros_like(mask).astype(int) - 1

        """ --- Adapted from https://stackoverflow.com/a/42812226/5166365 """
        mask_image = np.expand_dims(np.where(mask,1,0).astype(np.uint8),axis=2)

        num_blobs, markers, stats, _ = cv2.connectedComponentsWithStats(mask_image,connectivity = connectivity)

        markers-=1
        
        num_blobs-=1

        sizes = stats[:,-1]
 
        for blob_number in range(num_blobs):
            blob_label = blob_number
            if sizes[blob_number+1] >= min_pixels:
                result[markers == blob_label] = blob_label
        """ --- """
        
        return result

    @classmethod
    def bounds(cls, mask):
        nonzero_locations = np.transpose(np.nonzero(mask))
        min_x = np.min(nonzero_locations[:,1])
        min_y = np.min(nonzero_locations[:,0])
        max_x = np.max(nonzero_locations[:,1])
        max_y = np.max(nonzero_locations[:,0])
        return np.array((min_x, min_y),int), np.array((max_x, max_y),int)

    @classmethod
    def clipped_mask(cls, mask):
        tlc, brc = BooleanImageOps.bounds(mask)
        return np.copy(mask[tlc[1]:(brc[1]+1),tlc[0]:(brc[0]+1)])
        

    @classmethod
    def subject_boundary(cls, subject, target_segment_length=-1):
        subject_image = np.expand_dims(np.where(subject,1,0).astype(np.uint8),axis=2)
        contours,  hierarchy = cv2.findContours(subject_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            U.dprint("More than one contour was detected.")
            return None
        contour = np.squeeze(contours[0])
        if target_segment_length != -1:
            return Curve.resample_by_segment_length(contour,target_segment_length,True)
        return contour
    
    @classmethod
    def clean_by_pixel_count(cls,mask, fg_min_pixels, bg_min_pixels, connectivity=4):
        """!    Denoise a mask by removing small islands from foreground and background. Foreground is denoised first."""
        
        eps = 1 if connectivity == 4 else 2

        fg = mask.copy()
        U.dprint("Finding fg_grouping...")

        fg_grouping = BooleanImageOps.fast_filter_small_islands_and_get_grouping(mask,connectivity,fg_min_pixels)
        fg = fg_grouping != -1
        bg = np.logical_not(fg)
        U.dprint("Finding bg_grouping...")

        bg_grouping = BooleanImageOps.fast_filter_small_islands_and_get_grouping(bg,connectivity,fg_min_pixels)
        bg = bg_grouping != -1
        fg = np.logical_not(bg)

        return fg

    @classmethod
    def resize_to(cls,mask,W,H):
        mask_image = greyscale_plot_to_color_image(mask)
        resized_image = cv2.resize(mask_image,dsize=(W,H),interpolation=cv2.INTER_NEAREST)
        return resized_image[:,:,0] == 255

    @classmethod
    def scale_by(cls,mask,scale_factor):
        H, W = mask.shape
        return BooleanImageOps.resize_to(mask,int(W*scale_factor),int(H*scale_factor))

    @classmethod
    def where2D(cls,mask):
        """!    Returns list of positions of all true values in `mask`. 
                Positions are (x,y)
        """
        return np.flip(np.transpose(np.nonzero(mask)),axis=1)

class ImageStrip:

    def __init__(self,*images,**kwargs):

        self.images = copy.deepcopy(images)

        if "max_cols" in kwargs:
            self.max_cols = kwargs["max_cols"]
        else:
            self.max_cols = None

        if self.max_cols is None:
            self.max_cols = 1

    def getImagePixels(self):
        rows = []
        col_pointer = self.max_cols
        for image in self.images:
            if col_pointer == self.max_cols:
                rows.append([])
                col_pointer = 0
            rows[-1].append(image)
            col_pointer+=1
        if len(rows) > 1:
            while len(rows[-1]) < len(rows[-2]):
                rows[-1].append(np.zeros_like(rows[-2][0]))
        rows = [np.hstack(row) for row in rows]
        return np.vstack(rows)

    def saveSelf(self,path):
        PIL.Image.fromarray(self.getImagePixels()).save(path)

class Grouping:
    
    @classmethod
    def mix(cls,*groupings):
        H, W = groupings[0].shape
        result = np.zeros((H,W),int) - 1
        for grouping in groupings:
            mask = grouping!=-1
            result[mask] = grouping[mask]
        return result
    
    @classmethod
    def getIslands(cls,grouping):
        islands = {}
        unique_labels = np.unique(grouping)
        for label in unique_labels:
            if label == -1:
                continue
            islands[label] = np.flip(np.transpose(np.nonzero(grouping==label)),axis=1)
        return islands

    @classmethod
    def filter_small_islands_by_pixel_count(cls,grouping, min_pixel_count):
        islandsDict = Grouping.getIslands(grouping)
        new_grouping = grouping.copy()
        delete_list = []
        num_islands = len(islandsDict)
        for idx, (label, island) in enumerate(islandsDict.items()):
            print(f"Island {idx+1} of {num_islands}...")
            if len(island) < min_pixel_count:
                delete_list.append(label)
        for label in delete_list:
            new_grouping[grouping == label] = -1
        return new_grouping

    @classmethod
    def filter_all_but_k_largest_islands(cls,grouping,k):
        islandsDict = Grouping.getIslands(grouping)
        items = list(islandsDict.items())
        sorteditems = list(sorted(items,key=lambda item:len(item[1])))
        maintained_items = []
        num_maintained_items = min(k,len(sorteditems))
        for i in range(num_maintained_items):
            maintained_items.append(sorteditems[len(sorteditems)-1-i][0])
        new_grouping = grouping.copy()
        for label in np.unique(grouping):
            if label == -1:
                continue
            if label not in maintained_items:
                new_grouping[grouping == label] = -1
        return new_grouping

    @classmethod
    def watershed(cls, mask, seed_grouping, watershed_channel = None):
        """! function to easily employ cv2 watershed against a mask and image

        @param  mask                dictates the hard limits of the watershed fill
        @param  seed_grouping       a 2D int array used to generate the watershed hint. 
                                    Follows this library's convention and not opencv
        @param  watershed_channel   Allows for the specification of image or gradient information for watershed        
                                    If not supplied, it is set to a binary RGB image representing the mask parameter

        @return                     grouping after watershed 
        """

        if watershed_channel is None:
            watershed_channel = np.where(mask,1,0).astype(float)

        watershed_channel = np.squeeze(watershed_channel)
        if watershed_channel.ndim == 2:
            watershed_channel = greyscale_plot_to_color_image(watershed_channel)

        watershed_hint = np.zeros_like(seed_grouping) -1

        # Step 1, set seed_grouping background pixels to 1
        watershed_hint[seed_grouping==-1] = 1

        # Step 2, set seed group values
        watershed_hint[seed_grouping!=-1] = seed_grouping[seed_grouping!=-1] + 2

        # Step 3 set watershed targets
        watershed_hint[np.logical_and(seed_grouping==-1,mask)] = 0

        # Step 3.5 expand dims of watershed hint
        watershed_hint = np.expand_dims(watershed_hint,axis=2).astype(int) 

        # print(watershed_channel.shape,watershed_hint.shape)

        # Step 4 perform watershed
        result = np.squeeze(cv2.watershed(watershed_channel,watershed_hint))

        # Step 5, convert to this library's conventions
        ret_labels2D = np.zeros_like(seed_grouping) - 1

        ret_labels2D[result>=2] = result[result>=2] - 2

        ret_labels2D[~mask] = -1

        return ret_labels2D

    @classmethod
    def components_from_grouping(cls,grouping):
        return [grouping == label for label in range(np.max(grouping)+1)]

    @classmethod
    def get_edge_mask_from_grouping(cls,grouping,edge_thickness=1):
        """!    Uses a circular structuring element to find the edges"""

        element = circular_structuring_element(edge_thickness,int)
        element_num_pixels = np.count_nonzero(element)

        conv_result = scipy.ndimage.convolve(grouping,element,mode='constant',cval=-1)

        return conv_result!=(element_num_pixels*grouping)

    @classmethod
    def to_color_with_default_pallete(cls,grouping):
        pallete = Pallete()
        H, W = grouping.shape
        output_image = np.full((H,W,3),255, dtype=np.uint8)
        num_labels = np.max(grouping) + 1
        for label in range(num_labels):
            legend_color = pallete.nextColor()
            output_image[grouping==label,...] = legend_color
        return output_image
            
        


"""NOT TESTED!"""
class InfCanvas:

    def __init__(self):
        self.bounds = [None,None,None,None] #xmin xmax ymin ymax
        self.data = None

    def top_left_corner(self):
        return (self.bounds[0],self.bounds[2])

    def write_pixel(self,x,y,color):
        
        new_bounds = copy.deepcopy(self.bounds)
        new_bounds[0] = U.opt_min(x,self.bounds[0])
        new_bounds[1] = U.opt_max(x,self.bounds[1])
        new_bounds[2] = U.opt_min(y,self.bounds[2])
        new_bounds[3] = U.opt_max(y,self.bounds[3])

        if any([new_bounds[i]!=self.bounds[i] for i in range(4)]):
            if self.data is None:
                self.data = np.zeros((new_bounds[3]-new_bounds[2]+1,new_bounds[1]-new_bounds[0]+1,3),dtype=np.uint8)
            else:
                if new_bounds[0] < self.bounds[0]:
                    self.data = np.pad(self.data.copy(),[(0,0),(self.bounds[0]-new_bounds[0],0),(0,0)])
                if new_bounds[1] > self.bounds[1]:
                    self.data = np.pad(self.data.copy(),[(0,0),(0,new_bounds[1]-self.bounds[1]),(0,0)])
                if new_bounds[2] < self.bounds[2]:
                    self.data = np.pad(self.data.copy(),[(self.bounds[2]-new_bounds[2],0),(0,0),(0,0)])
                if new_bounds[3] > self.bounds[3]:
                    self.data = np.pad(self.data.copy(),[(0,new_bounds[3]-self.bounds[3]),(0,0),(0,0)])
        self.bounds = new_bounds
        cx, cy = self.top_left_corner()
        self.data[y-cy, x-cx] = color 

    def write_image(self,x,y,image):
        for dx in range(image.shape[1]):
            for dy in range(image.shape[0]):
                self.write_pixel(x+dy,y+dy,image[dy,dx])
                

    def getImagePixels(self):
        return self.data.copy()

import numba.types
import numba.experimental

spec = [
    ('W',numba.types.int64),
    ('H',numba.types.int64),
    ('data',numba.types.uint8[:,:,:])
]

class WindowableArray:
    
    @classmethod
    def axis_get_overlap(cls,x,w,W):
        
        big_start_idx = x
        big_end_idx = x + w
        
        original_big_start_idx = big_start_idx
        original_big_end_idx = big_end_idx
        
        if big_start_idx < 0:
            big_start_idx = 0
        if big_end_idx > W:
            big_end_idx = W
            
        small_padding_start = big_start_idx - original_big_start_idx
        small_padding_end = big_end_idx - original_big_end_idx
        
        small_start_idx = small_padding_start
        small_end_idx = w-small_padding_end
        
        return SNs(
            image = SNs(
                start = big_start_idx,
                end = big_end_idx
            ),
            kernel = SNs(
                start = small_start_idx,
                end = small_end_idx
            )
        )
        
    @classmethod
    def axes_get_overlap(cls, x,y,w,h,W,H):
        
        overlap_x = WindowableArray.axis_get_overlap(x,w,W)
        overlap_y = WindowableArray.axis_get_overlap(y,h,H)
        
        image_slice = np.s_[
            overlap_y.image.start:overlap_y.image.end,
            overlap_x.image.start:overlap_x.image.end,
            ...]
        
        kernel_slice = np.s_[
            overlap_y.kernel.start:overlap_y.kernel.end,
            overlap_x.kernel.start:overlap_x.kernel.end,
            ...]
        
        return image_slice, kernel_slice
    
    @classmethod
    def detect_trivial_case(cls,x,y,w,h,W,H):
        if x < 0 and x+w <= 0:
            return True
        if x>=W and x+w >= W:
            return True
        if y < 0 and y+h <= 0:
            return True
        if y>=H and y+h >= H:
            return True
        return False
        
    def __init__(self,data):
        self.data = data.copy()
        self.H, self.W = data.shape[:2]
        if data.ndim ==2:
            self.nchannels = 0
        else:
            self.nchannels = data.shape[-1]
            
    def is_in_bounds(self,x,y):
        return x>=0 and x<self.W and y>=0 and y<self.H
    
    def set_item(self,x,y,value):
        self.data[y,x,...] = value
    
    def get_item(self,x,y,value):
        return self.data[y,x]
    
    def set_window(self,x,y,w,h,values):
        if WindowableArray.detect_trivial_case(x,y,w,h,self.W,self.H):
            return
        image_slice, kernel_slice = WindowableArray.axes_get_overlap(x,y,w,h,self.W,self.H)
        self.data[image_slice] = values[kernel_slice]
        
    def get_window(self,x,y,w,h,fill_value=0):
        if WindowableArray.detect_trivial_case(x,y,w,h,self.W,self.H):
            return np.full((h,w)+tuple() if self.nchannels == 0 else (self.nchannels,),fill_value).astype(self.data.dtype)
        image_slice, kernel_slice = WindowableArray.axes_get_overlap(x,y,w,h,self.W,self.H)
        return self.data[image_slice].copy()

@numba.experimental.jitclass(spec)
class WindowableRGBImage:

    def __init__(self,W,H):
        self.W = W
        self.H = H
        self.data = np.zeros((H,W,3),dtype=np.uint8)
    
    def point_is_in_bounds(self,x,y):
        return x>=0 and x<self.W and y>=0 and y < self.H

    def write_pixel(self,x,y,color):
        if not self.point_is_in_bounds(x,y):
            return
        for c in range(3):
            self.data[y,x,c] = color[c]

    def read_pixel(self,x,y):
        if not self.point_is_in_bounds(x,y):
            return np.zeros((3,),dtype=np.uint8)
        return np.array([
            self.data[y,x,0],
            self.data[y,x,1],
            self.data[y,x,2]
        ],dtype=np.uint8)

    def read_window(self,x,y,w,h):
        result = np.zeros((h,w,3),dtype=np.uint8)
        for dx in range(w):
            for dy in range(h):
                pixel = self.read_pixel(x+dx,y+dy)
                for c in range(3):
                    result[dy,dx,c] = pixel[c]
        return result

    def read_window_with_default(self,x,y,w,h,color):
        result = np.zeros((h,w,3),dtype=np.uint8)
        for dx in range(w):
            for dy in range(h):
                if self.point_is_in_bounds(x+dx,y+dy):
                    pixel = self.read_pixel(x+dx,y+dy)
                else:
                    pixel=color
                for c in range(3):
                    result[dy,dx,c] = pixel[c]
        return result

    def write_window(self,x,y,w,h,data):
        for dx in range(w):
            for dy in range(h):
                self.write_pixel(x+dx,y+dy,np.array([
                    data[dy,dx,0],
                    data[dy,dx,1],
                    data[dy,dx,2]
                ],dtype=np.uint8))

    def write_image(self,x,y,image):
        h, w = image.shape[0], image.shape[1]
        self.write_window(x,y,w,h,image)

    def getImagePixels(self):
        return self.data.copy()

def to_WindowableRGBImage(image):
    H, W = image.shape[:2]
    windowable_image = WindowableRGBImage(W,H)
    windowable_image.write_image(0,0,image)
    return windowable_image

class Curve:

    @classmethod
    def visualize_curves(cls,W,H,curves,bgcolor=(255,255,255),T=1):
        pallete = Pallete()
        image = np.full((H,W,3),bgcolor,dtype=np.uint8)
        for i, curve in enumerate(curves):
            curve_color = pallete.nextColor()
            image = Image.draw_curve_on(image,curve,curve_color,T)
        return image
        

    @classmethod
    def measure_length(cls,curve, closed=False):
        total_length = 0

        for i in range(len(curve)-1 if not closed else len(curve)):
            ia = i
            ib = (i+1) % len(curve)
            total_length += np.linalg.norm(curve[ib].astype(float)-curve[ia].astype(float))

        return total_length

    @classmethod
    def polynomial_best_fit_curve(cls, contour, min_degree = 0, max_degree=20):

        x_values = [float(contour_point[0]) for contour_point in contour]
        y_values = [float(contour_point[1]) for contour_point in contour]
        t_values = np.linspace(0.0,1.0,len(contour))

        items_and_metrics = []

        for degree in range(min_degree,max_degree+1):
            coeffs_x = np.polyfit(t_values,x_values,degree)
            coeffs_y = np.polyfit(t_values,y_values,degree)
            predicted_x = np.polyval(coeffs_x,t_values)
            predicted_y = np.polyval(coeffs_y,t_values)
            R2X = U.coefficient_of_determination(x_values,predicted_x)
            R2Y = U.coefficient_of_determination(y_values,predicted_y)
            MEAN_R2 = 0.5 * R2X + 0.5 * R2Y
            items_and_metrics.append((np.array(list(zip(predicted_x,predicted_y)),contour.dtype),MEAN_R2))

        items_and_metrics = sorted(items_and_metrics,key=lambda entry: entry[1])

        return items_and_metrics[-1][0]

    @classmethod
    def resample_by_segment_length(cls, curve, segment_length, closed=False):

        curve_length = Curve.measure_length(curve)
        num_segments = int(math.ceil(curve_length / segment_length))

        shape = None

        if closed:
            shape = shapely.geometry.polygon.LinearRing(np.array(curve,float))
        else:
            shape = shapely.geometry.linestring.LineString(np.array(curve,float))

        new_curve = []

        for i in range(num_segments):
            progress = (i+1)/num_segments
            interpolated_point = shape.interpolate(progress,True)
            new_curve.append(np.array(list(interpolated_point.coords)[0],float))

        return np.array(new_curve,float)

        
    @classmethod
    def fit_univariate_spline(cls,curve):
        t_values = np.linspace(0.0,1.0,len(curve))
        x_values = np.array([pt[0] for pt in curve],float)
        y_values = np.array([pt[1] for pt in curve],float)
        x_fitter = scipy.interpolate.UnivariateSpline(t_values,x_values)
        y_fitter = scipy.interpolate.UnivariateSpline(t_values,y_values)
        predicted_x = x_fitter(t_values)
        predicted_y = y_fitter(t_values)
        return np.array(list(zip(predicted_x,predicted_y)),float)

    @classmethod
    def curve_to_mask(cls,curve,W=None,H=None):

        # @TODO: double-check this function

        llc = np.array([0,0],float)

        if H is None:
            if W is not None:
                H=W
            else:
                bounds = np.min(curve[:,0]),np.min(curve[:,1]),np.max(curve[:,0]),np.max(curve[:,1])
                llc=np.array([bounds[0],bounds[1]],float) 
                H = int(bounds[3]-bounds[1]+1)
                W = int(bounds[2]-bounds[0]+1)
        polygon = shapely.geometry.polygon.Polygon(np.array(curve,float)-llc)
        mask = rasterio.features.rasterize([polygon],out_shape=(H,W))
        return mask

    @classmethod
    def curve_tangents_and_normals(cls,curve):
        difference_vectors = []
        for i in range(len(curve)-1):
            difference_vectors.append(np.array(curve[i+1],float)-np.array(curve[i],float))
        difference_vectors.append(difference_vectors[-1])
        return np.array(difference_vectors,float),np.array([(-pt[1],pt[0]) for pt in difference_vectors],float)

class Draw:

    @classmethod
    def draw_centered_text(cls,image, message,x,y,scale,color,thickness):
        textSize, baseline = cv2.getTextSize(message,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
        textSize = list(textSize)
        adj_x = int(x-textSize[0]/2)
        adj_y = int(y+textSize[1]/2)
        new_image = cv2.putText(image,message,(adj_x,adj_y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness)
        return new_image

    @classmethod
    def draw_annotation(cls,image,annotation,x,y,text_scale,color,text_thickness,radius_scale_factor=1.0,circle_thickness=None):
        if circle_thickness is None:
            circle_thickness = text_thickness
        print(text_scale)
        textSize,_ = cv2.getTextSize(annotation,cv2.FONT_HERSHEY_SIMPLEX,text_scale,text_thickness)
        textSize = list(textSize)
        radius = int(radius_scale_factor * np.max(textSize))
        new_image =image.copy()
        new_image = cv2.circle(new_image,(x,y),radius,(255,255,255),-1)
        new_image = cv2.circle(new_image,(x,y),radius,(255,255,255),circle_thickness)
        new_image = Draw.draw_centered_text(new_image,annotation,x,y,text_scale,color,text_thickness)
        return new_image

    @classmethod
    def line_mask(cls,A,B,W,H,T=1):
        arr = np.zeros((H,W,3))
        arr=cv2.line(arr,A.astype(int),B.astype(int),(255,255,255),T)
        arr = U.npAndMany(arr[:,:,0] > 0,arr[:,:,1] > 0,arr[:,:,2] > 0)
        return arr

class Image:

    @classmethod
    def select_where_same(cls, image, target_color):
        return U.npAndMany(*[image[:,:,c]==target_color[c] for c in range(3)])
        
    @classmethod
    def select_where_different(cls, image, target_color):
        return U.npOrMany(*[image[:,:,c]!=target_color[c] for c in range(3)])
        

    @classmethod
    def mix_on_background(images,bgcolor=(0,0,0)):
        assert all([images[i].shape==images[0].shape for i in range(len(images))])
        H, W = images[0].shape[:2]
        result = np.full((H,W,3),bgcolor,dtype=np.uint8)
        for image in images:
            selection = Image.select_where_equal(image,bgcolor)
            fg = Image.select_where_different(image,bgcolor)
            result[fg,:] = image[fg,:]

    @classmethod
    def mix(cls,images,weights):
        final_image = np.zeros_like(images[0]).astype(float)
        for image, weight in zip(images,weights):
            final_image+=weight*image
        final_image = np.clip(final_image,0,255)
        return final_image.astype(np.uint8)

    @classmethod
    def binary_mix(cls,background,foreground,mask):
        """! Mix a foreground and background image according to foreground mask

        @param  background   background image -- np.array, shape=(H,W,3), dtype=np.uint8
        @param  foreground   foreground image -- np.array, shape=(H,W,3), dtype=np.uint8
        @param  mask         foreground mask -- np.array, shape=(H,W) dtype= bool, int, or np.uint8 
        """

        mask_3d = np.dstack((mask,)*3)

        return np.where(mask_3d,foreground,background).astype(np.uint8)
    
    @classmethod
    def draw_curve_on(cls,image,curve,color=(255,255,255),T=1,closed=False):
        image_copy = image.copy()
        mask = np.zeros(image.shape[:2],dtype=bool)
        mask = BooleanImageOps.draw_curve_on(mask,curve,T,closed)
        mask_image = BooleanImageOps.boolean_to_color(mask,color)
        return Image.binary_mix(image_copy,mask_image,mask)
    
    