import os
from time import sleep
import pickle

from core.file_picker import askopenfilename
from core.io import imload, imsave
from rpe_cell_counting import get_conservative_cell_mask
from core.python_utils import fluiddict

import eel
import orjson
import numpy as np

FS_POLL_TIME = 0.125

MARKER_RADIUS=2

DELETE_RADIUS = MARKER_RADIUS * 2

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
    print(pixels.shape) #@delete
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
    conservative_cell_mask = get_conservative_cell_mask(pixels)
    
    DARK_FRACTION = 0.25 # The opacity of the original image in the non-mask regions
    
    background = np.zeros_like(pixels)
    
    new_pixels = np.zeros_like(pixels)
    
    for c in range(3):
        new_pixels[:,:,c] = np.where(conservative_cell_mask,pixels[:,:,c],(1.0-DARK_FRACTION)*background[:,:,c] + DARK_FRACTION*pixels[:,:,c]) 
        
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


eel.start("index.html")