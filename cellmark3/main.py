import os
from time import sleep
import pickle
from math import floor
from termcolor import colored

os.system("color")

from core.file_picker import askopenfilename
from core.io import imload, imsave
from rpe_cell_counting import get_conservative_cell_mask
from core.python_utils import fluiddict, rescale_array
from core.image_ops import circular_structuring_element, draw_element_on, voronoi, connected_components, distance_transform_L1, unique_quantize, binarize_otsu, get_outer_contour, check_if_contour_bounds_pixels_in_mask, resample_contour_by_segment_length
from core.array_utils import bin_edges_to_midpoints
from core.debugging import imshow
from core.array_utils import multiply_arrays
from core.data_analysis import find_threshold_minimizing_intra_class_variance

import eel
import orjson
import numpy as np
import cv2
import matplotlib.pyplot as plt

FS_POLL_TIME = 0.125

PREFILL_GROUP_MIN_PIXELS = 7 * 7

PREFILL_CONTOUR_RESAMPLE_PIXELS = 5

MARKER_RADIUS=2

DELETE_RADIUS = MARKER_RADIUS * 2

MARKER_ELEMENT = circular_structuring_element(MARKER_RADIUS, dtype=bool)

eel.init("public")

def pixels_to_json(pixels):
    return orjson.dumps(
        {
            "w":pixels.shape[1],
            "h":pixels.shape[0],
            "nchannels":pixels.shape[2],
            "data":[float(value) for value in np.ravel(pixels)]
        }
    ).decode('utf-8')
    
def json_to_pixels(imageJSON):
    imageObj = fluiddict(**(orjson.loads(imageJSON)))
    pixels = np.ravel(np.array(imageObj.data,dtype=np.uint8)).reshape((imageObj.h, imageObj.w, imageObj.nchannels))
    return pixels
    
@eel.expose
def get_marker_radius():
    return MARKER_RADIUS    

@eel.expose
def get_delete_radius():
    return DELETE_RADIUS   
    
@eel.expose
def pick_image(): #ionly returns the path, will be loaded ina seperate step
    return askopenfilename()

@eel.expose
def load_image(path):
    pixels=imload(path)
    return pixels_to_json(pixels)

def process_group_mask_for_group_number(cluster_map, group_number):
    
        group_mask = cluster_map == group_number
        
        if np.count_nonzero(group_mask) < PREFILL_GROUP_MIN_PIXELS:
            print("Pixel group too small.")
            return None
        
        try:
        
            outer_contour = get_outer_contour(group_mask)
            
            if not check_if_contour_bounds_pixels_in_mask(outer_contour, group_mask):
                print("Pixel group had malformed border")
                return None
        
        except Exception as e:
            print(e)
            return None
            
        outer_contour = resample_contour_by_segment_length(outer_contour, PREFILL_CONTOUR_RESAMPLE_PIXELS, closed=True)
        
        centroid_rc = np.mean((np.transpose(np.nonzero(group_mask))).astype(float), axis=0)
        
        centroid_xy = np.flip(centroid_rc)
        
        distances_to_centroid = []
        
        for contour_point in outer_contour:
            distances_to_centroid.append(
                np.linalg.norm(
                    centroid_xy.astype(float) -
                    contour_point.astype(float)
                )
            )
            
        border_variation = np.var(distances_to_centroid)/((np.mean(distances_to_centroid))**2) * float(np.count_nonzero(group_mask))
        
        return border_variation

def cluster_filter_pass(pixels, cluster_map):
    
    num_groups = np.max(cluster_map) + 1

    border_variations = {}

    for group_number in range(num_groups):
        
        border_variation = process_group_mask_for_group_number(cluster_map, group_number)
        
        if border_variation is not None:
        
            border_variations[group_number] = border_variation
    
    border_variation_data = np.array(list(border_variations.values()),float)
    
    threshold = find_threshold_minimizing_intra_class_variance(border_variation_data)

    new_cluster_map = cluster_map.copy()
    
    new_pixels = pixels.copy()
    
    for group_number in range(num_groups):

        if group_number not in border_variations:
            
            new_cluster_map[cluster_map==group_number]=-1
            
            new_pixels[cluster_map==group_number, ...] = (255,255,255)
            
        else:
            
            if border_variations[group_number] < threshold:
                
                new_pixels[cluster_map==group_number, ...] = (255,0,255)
                
            else:
            
                new_pixels[cluster_map==group_number, ...] = (0,255,0)
            
                new_cluster_map[cluster_map==group_number] = -1
                
    return new_pixels, new_cluster_map

def get_random_colors(num_colors):
    random_colors = [(255,255,255)]
    for _ in range(num_colors):
        random_colors.append((255*np.array([
            np.random.uniform(),
            np.random.uniform(),
            np.random.uniform()
            ],float)).astype(np.uint8))
    return np.array(random_colors,np.uint8)

@eel.expose
def load_image_rpe(path):
    
    pixels=imload(path).copy()

    conservative_cell_mask = get_conservative_cell_mask(pixels)
    
    cluster_map = connected_components(conservative_cell_mask)
    
    new_pixels, new_cluster_map = pixels.copy(), cluster_map.copy()

    for _ in range(5):    
        new_pixels, new_cluster_map = cluster_filter_pass(new_pixels, new_cluster_map)
        
    single_cell_masses = []    
        
    num_groups = int(np.max(new_cluster_map)) + 1
    
    for group_number in list(np.unique(new_cluster_map)):
        
        if group_number == -1:
            continue

        single_cell_masses.append(np.count_nonzero(new_cluster_map==group_number))
                
    median = np.median(single_cell_masses)
    
    print(single_cell_masses)    
    
    print(median)        
                
    random_colors = get_random_colors(int(np.max(cluster_map))+1) # Will never need more than this many colors. Will likely need MUCH less
     
    num_groups = int(np.max(cluster_map)) + 1            
  
    new_pixels = pixels.copy()

    for group_number in range(num_groups):
        
        mass = np.count_nonzero(cluster_map==group_number)

        relative_mass = int(round(mass/median))
        
        # print(median,np.all(cluster_map[cluster_map==group_number]==new_cluster_map[cluster_map==group_number]),mass,relative_mass)
        
        new_pixels[cluster_map==group_number, ...] = random_colors[relative_mass]            
                
    return pixels_to_json(new_pixels) 

@eel.expose
def get_conservative_cell_mask_rpe(path):
    pixels=imload(path)
    conservative_cell_mask = get_conservative_cell_mask(pixels)
    return pixels_to_json(
        np.dstack((
            np.where(conservative_cell_mask,255,0)
            ,)*3)
    ) 
    
@eel.expose
def save_markings(imageFilepath,imageJSON, annotatedImageJSON, markingsJSON):
    
    save_parent_directory = os.path.realpath(os.path.dirname(imageFilepath))
    
    image_basename = os.path.basename(imageFilepath)
    
    save_directory = os.path.join(save_parent_directory,image_basename+".annotated")
    
    if not os.path.isdir(save_directory):
        
        os.makedirs(save_directory,exist_ok=True)
        
        while not os.path.isdir(save_directory):
            sleep(FS_POLL_TIME)
    
    vis_pixels = json_to_pixels(imageJSON)
    
    pixels = imload(imageFilepath)
    
    annotated_pixels = json_to_pixels(annotatedImageJSON)
    
    imsave(pixels,os.path.join(save_directory,image_basename))
    
    imsave(vis_pixels,os.path.join(save_directory,"vis."+image_basename))
    
    imsave(vis_pixels,os.path.join("temp","vis."+image_basename))
    
    imsave(annotated_pixels,os.path.join(save_directory,"annotated."+image_basename))
    
    markings = orjson.loads(markingsJSON)
    
    """ Excerpt from: D:\SwamynathanLab\Cell Marking Software -- Stable\ImageJCloneWithPython\main.py
    
    save_state = {}
    save_state["marker_positions"] = marker_positions
    save_state["active_image"] = active_image
    save_state["active_image_path"] = active_image_path
    save_state["active_sf"] = active_sf
    with open(save_dir+'save_state.markcells.pkl','wb') as fl:
        pickle.dump(save_state,fl) 

    """
    
    save_state = {}
    save_state["marker_positions"] = markings
    save_state["active_image"] = pixels
    save_state["active_image_path"] = os.path.join(save_directory,image_basename)
    save_state["active_sf"] = 1.0
    save_state["vis_pixels"] = vis_pixels
    save_state["annotated_image"] = annotated_pixels
    save_state["marker_radius"] = MARKER_RADIUS
    
    
    with open(os.path.join(save_directory,"save_state.markcells.pkl"),"wb") as fl:
        pickle.dump(save_state,fl)
        
    stats_text = f"""
Cell Count: {len(markings)}
    
    """
    
    with open(os.path.join(save_directory, "stats.txt"), "w") as fl:
        fl.write(stats_text)

@eel.expose
def load_markings(imageFilepath):
    
    save_parent_directory = os.path.realpath(os.path.dirname(imageFilepath))
    
    image_basename = os.path.basename(imageFilepath)
    
    save_directory = os.path.join(save_parent_directory,image_basename+".annotated")
    
    if not os.path.isdir(save_directory):
        return "no_session"
    
    if not os.path.isfile(os.path.join(save_directory,"save_state.markcells.pkl")):
        return "no_session"
    
    save_state = {}
    
    with open(os.path.join(save_directory,"save_state.markcells.pkl"),"rb") as fl:
        save_state = pickle.load(fl)
        
    response = fluiddict()
    
    response.image = pixels_to_json(save_state["vis_pixels"])
    response.annotatedImage = pixels_to_json(save_state["annotated_image"])
    response.marks = save_state["marker_positions"]
    
    return orjson.dumps(response.datastore).decode("utf-8")



def adjust_markings_pass(method, markings, image, cell_mask):

    cell_mask_image = np.dstack((np.where(cell_mask,255,0),)*3).astype(np.uint8)

    image_LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    L, A, B = image_LAB[:,:,0].astype(float), image_LAB[:,:,1].astype(float), image_LAB[:,:,2].astype(float)
    
    L = rescale_array(L)
    
    A = rescale_array(A)
    
    B = rescale_array(B)

    L_image = np.dstack((255*L,)*3).astype(np.uint8)
    A_image = np.dstack((255*A,)*3).astype(np.uint8)
    B_image = np.dstack((255*B,)*3).astype(np.uint8)

    imsave(L_image,"temp/L.png")
    imsave(A_image,"temp/A.png")
    imsave(B_image,"temp/B.png")
    
    # dist = distance_transform_L1(cell_mask).astype(float)
    
    # adjusted_dist = np.zeros_like(dist)
    
    # cluster_map = connected_components(cell_mask)
    
    # for group_number in range(np.max(cluster_map) + 1):
        
    #     island_rc = np.transpose(np.nonzero(cluster_map == group_number))
        
    #     distances = [dist[r,c] for r,c in island_rc]
        
    #     max_distance = np.max(distances)
        
    #     for r, c in island_rc:
    #         adjusted_dist[r,c] = dist[r,c] / max_distance
            
    # imsave(np.dstack((255*adjusted_dist,)*3).astype(np.uint8),"temp/adjusted_dist.png")
    
    # combined = multiply_arrays(B,1-A,1-L)
    
    combined = 1-L
    
    combined[np.logical_not(cell_mask)] = 0
    
    combined_image = np.dstack((255*rescale_array(combined),)*3).astype(np.uint8)
    
    imsave(combined_image,"temp/combined.png")
    
    hint = np.zeros(image.shape[:2], int) -1
  
    num_marks = len(markings)
  
    for i, mark in enumerate(markings):
        
        print(f"Drawing marking {i+1} of {num_marks}...")
        
        hint = draw_element_on(hint,mark[0],mark[1],MARKER_ELEMENT,i+1)
        
    unseeded_area = np.logical_and(cell_mask, hint==-1)
    
    hint[unseeded_area] = 0

    # still need to decide on the best image to use    
    # watershed_image = np.dstack((255*(1.0-L),)*3).astype(np.uint8)
    watershed_image = combined_image
    
    watershed_labels = voronoi(hint) if method=="voronoi" else cv2.watershed(watershed_image, hint).squeeze()
    
    watershed_labels[watershed_labels==0]=-1 # just in case

    watershed_labels[watershed_labels>=1]-=1
    
    new_markings = []
    
    num_groups = np.max(watershed_labels) + 1
    
    cluster_map = connected_components(cell_mask)
    
    for group_number in range(num_groups):
        
        island_rc = np.transpose(np.nonzero(watershed_labels==group_number))
        
        weights = np.array([combined[r,c] for r,c in island_rc],float)
        
        weights/= np.sum(weights)
        
        centroid = (np.sum(
            
                            np.multiply(
                                island_rc.astype(float),
                                np.stack((weights,)*2,axis=1)
                            )
                           
                            ,axis=0)).astype(int)
        
        marker = markings[group_number]
        
        marker = [int(floor(marker[0])), int(floor(marker[1]))]

        result_marker = np.flip(centroid)

        if centroid[0] < 0 or centroid[0] >= cell_mask.shape[0] or centroid[1] < 0 or centroid[1] >= cell_mask.shape[1]:
            result_marker = marker
        
        elif not cell_mask[centroid[0],centroid[1]]:
            result_marker = marker
        
        elif cluster_map[centroid[0],centroid[1]] != cluster_map[marker[1],marker[0]]:
            result_marker = marker
        
        new_markings.append(result_marker) 
    
    return new_markings
    

@eel.expose
def auto_adjust_markings(imageJSON, cellMaskJSON, markingsJSON):
    cell_mask = json_to_pixels(cellMaskJSON)[:,:,0] > 0
    image = json_to_pixels(imageJSON)
    markings = orjson.loads(markingsJSON)
    
    markings = adjust_markings_pass("voronoi",markings,image,cell_mask)
    # markings = adjust_markings_pass("watershed",markings,image,cell_mask)
    
    result = [[float(mark[0]),float(mark[1])] for mark in markings]
    
    return orjson.dumps(result).decode('utf-8')

@eel.expose
def auto_preload_markings(imageJSON, cellMaskJSON):
    
    image = json_to_pixels(imageJSON)
    
    cell_mask_image = json_to_pixels(cellMaskJSON)
    
    cell_mask = cell_mask_image[:,:,0] > 0
    
    cell_mask_image = np.dstack((np.where(cell_mask,255,0),)*3).astype(np.uint8)

    image_LAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    L, A, B = image_LAB[:,:,0].astype(float), image_LAB[:,:,1].astype(float), image_LAB[:,:,2].astype(float)
    
    L = rescale_array(L)
    
    A = rescale_array(A)
    
    B = rescale_array(B)

    L_image = np.dstack((255*L,)*3).astype(np.uint8)
    A_image = np.dstack((255*A,)*3).astype(np.uint8)
    B_image = np.dstack((255*B,)*3).astype(np.uint8)

    imsave(L_image,"temp/L.png")
    imsave(A_image,"temp/A.png")
    imsave(B_image,"temp/B.png")
     
    # dist = distance_transform_L1(cell_mask).astype(float)
    
    # adjusted_dist = np.zeros_like(dist)
    
    # cluster_map = connected_components(cell_mask)
    
    # for group_number in range(np.max(cluster_map) + 1):
        
    #     island_rc = np.transpose(np.nonzero(cluster_map == group_number))
        
    #     distances = [dist[r,c] for r,c in island_rc]
        
    #     max_distance = np.max(distances)
        
    #     for r, c in island_rc:
    #         adjusted_dist[r,c] = dist[r,c] / max_distance
            
    # imsave(np.dstack((255*adjusted_dist,)*3).astype(np.uint8),"temp/adjusted_dist.png")
    
    # combined = multiply_arrays(B,1-A,1-L)
    
    combined = 1 - L
    
    combined[np.logical_not(cell_mask)] = 0
    
    combined_image = np.dstack((255*rescale_array(combined),)*3).astype(np.uint8)
    
    imsave(combined_image,"temp/combined.png")
    
    print("quantizing...")
    
    thresholded = binarize_otsu(rescale_array(combined))
    
    cluster_map = connected_components(thresholded)

    result = []
    
    for group_number in range(np.max(cluster_map) + 1):
        
        island_rc = np.transpose(np.nonzero(cluster_map==group_number))
        
        weights = np.array([combined[r,c] for r,c in island_rc],float)
        
        weights/= np.sum(weights)
        
        centroid = (np.sum(
            
                            np.multiply(
                                island_rc.astype(float),
                                np.stack((weights,)*2,axis=1)
                            )
                           
                            ,axis=0)).astype(int)
        
        marker = None
        
        result_marker = np.flip(centroid)

        if centroid[0] < 0 or centroid[0] >= cell_mask.shape[0] or centroid[1] < 0 or centroid[1] >= cell_mask.shape[1]:
            result_marker = marker
        
        elif not cell_mask[centroid[0],centroid[1]]:
            result_marker = marker
        
        if result_marker is not None:
        
            result.append(result_marker) 
    
    return orjson.dumps([
        
        [float(mark[0]),float(mark[1])] for mark in result
        
        ]).decode("utf-8")
    
    
    

eel.start("index.html")