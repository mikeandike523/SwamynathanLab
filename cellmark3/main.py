import os
from time import sleep
import pickle

from core.file_picker import askopenfilename
from core.io import imload, imsave
from rpe_cell_counting import get_conservative_cell_mask
from core.python_utils import fluiddict
from core.image_ops import circular_structuring_element, draw_element_on
from core.debugging import imshow

import eel
import orjson
import numpy as np
import cv2


FS_POLL_TIME = 0.125

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

@eel.expose
def load_image_rpe(path):
    pixels=imload(path)

    # # Remove this feature, may be causing unnecessary bias
    #DARK_FRACTION = 0.25 # The opacity of the original image in the non-mask regions
    #conservative_cell_mask = get_conservative_cell_mask(pixels)
    # background = np.zeros_like(pixels)
    # new_pixels = np.zeros_like(pixels)
    # for c in range(3):
    #     new_pixels[:,:,c] = np.where(conservative_cell_mask,pixels[:,:,c],(1.0-DARK_FRACTION)*background[:,:,c] + DARK_FRACTION*pixels[:,:,c]) 
    
    new_pixels = pixels.copy()
        
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

@eel.expose
def auto_adjust_markings(imageJSON, cellMaskJSON, markingsJSON):
    
    cell_mask = json_to_pixels(cellMaskJSON)[:,:,0] > 0
    
    image = json_to_pixels(imageJSON)
    
    markings = orjson.loads(markingsJSON)
    
    hint = np.zeros(image.shape[:2], int) -1
  
    num_marks = len(markings)
  
    for i, mark in enumerate(markings):
        
        print(f"Drawing marking {i+1} of {num_marks}...")
        
        hint = draw_element_on(hint,mark[0],mark[1],MARKER_ELEMENT,i+1)
        
    
    unseeded_area = np.logical_and(cell_mask, hint==-1)
    

    hint[unseeded_area] = 0
    
    watershed_labels = cv2.watershed(image, hint).squeeze()
    
    watershed_labels[watershed_labels==0]=-1 # just in case

    watershed_labels[watershed_labels>=1]-=1
    
    new_markings = []
    
    num_groups = np.max(watershed_labels) + 1
    
    for group_number in range(num_groups):
        
        island_rc = np.transpose(np.nonzero(watershed_labels==group_number))
        
        centroid = (np.mean(island_rc.astype(float),axis=0)).astype(int)
        
        if not cell_mask[centroid[0],centroid[1]]:
            continue
    
        new_markings.append(np.flip(centroid)) 

    result = [[float(mark[0]),float(mark[1])] for mark in new_markings]
    
    return orjson.dumps(result).decode('utf-8')

eel.start("index.html")