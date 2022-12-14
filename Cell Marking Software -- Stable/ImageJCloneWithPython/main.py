import json
import time
import eel
import numpy as np 
import PIL.Image
import cv2
import sys
from collections import deque
from threading import Thread
import functools
import copy
import pickle
import simplefilepicker
import os
import shutil

INTERPOLATION_MODE = cv2.INTER_CUBIC

sys.path.insert(0,".")
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+"\\TensorflowCellSeperationExperiments\\src")
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+"\\Retina Mapping Project")


import image_processing as IP
import utils as U

from core.types import fluiddict

from fancyfilepicker import FancyFilePickerApplication
from plugins.mark_cells.masking import get_rpe_nucleus_mask
from plugins.mark_cells.watershed import perform_watershed

eel.init('public')

active_image = None
active_image_path=None
active_sf = 1.0

active_background_mask = None

messages = deque([])

active_plugin = None

CONSOLE_NUM_ROWS = 10

VERTICAL_MIN_RATIO = 2.50



@eel.expose
def get_VERTICAL_MIN_RATIO():
    return VERTICAL_MIN_RATIO

def plugin_console_log(message):
    global messages
    messages.append(message)
    if len(messages) > CONSOLE_NUM_ROWS:
        messages.popleft()

def image_to_json_compatible_obj(img):
    H, W = img.shape[:2]
    # img_copy = img.copy()
    # img_copy = img_copy[:,:,:3]
    data = list(np.ravel(img).astype(float))
    return {
        "W":W,
        "H":H,
        "data":data
    }

def image_to_json(img):
    H, W = img.shape[:2]
    # img_copy = img.copy()
    # img_copy = img_copy[:,:,:3]
    data = list(np.ravel(img).astype(float))
    return json.dumps({
        "W":W,
        "H":H,
        "data":data
    })

def load_image(fp,sf,set_active_image=False):

    global active_image
    global active_image_path
    global active_background_mask
    global active_sf
    global active_plugin


    if fp=="" and sf == 0:
        return json.dumps("none")
    img = np.asarray(PIL.Image.open(fp))
    img = cv2.resize(img,dsize=(int(sf*img.shape[1]),int(sf*img.shape[0])),interpolation=INTERPOLATION_MODE)

    #change black background to white
    # black_pixels_mask = functools.reduce(np.logical_and,[img[:,:,c] == 0 for c in range(3)])
    # BG_MIN_PIXELS = 500
    # grouping = IP.BooleanImageOps.connected_components(black_pixels_mask,8)
    # grouping = IP.Grouping.filter_small_islands_by_pixel_count(grouping,BG_MIN_PIXELS)
    # black_pixels_mask = grouping!=-1
    # img[black_pixels_mask,...] = 255

    # active_background_mask = black_pixels_mask

    if set_active_image:
        active_image = img[:,:,:3] 
        active_image_path = fp
        active_sf = sf
        active_plugin = None
            
    return image_to_json(active_image)

@eel.expose
def load_image_by_path(fp,sf):
    return load_image(fp,sf,True)

@eel.expose
def pick_file():

    global active_plugin
    active_plugin = None


    print("Opening File Picker...")
    result = None
    def callback(fp, sf):
        nonlocal result
        result = load_image(fp,sf,True)
    FancyFilePickerApplication().open(callback)
    while result is None:
        time.sleep(0.25)
    return result

active_cell_mask = None


""" --- Cell Marking Plugin --- """

# Store the state of the mark-cells editor

mark_cells_editor_state = fluiddict(default_factory=None)

marker_positions = []

MARKER_RADIUS=3
MARKER_DELETE_RADIUS=MARKER_RADIUS*1.5
DELETE_MARK_NUM_NEIGHBORS = 10

marker_element = IP.circular_structuring_element(MARKER_RADIUS,bool)
marker_element_xy = IP.BooleanImageOps.where2D(marker_element)

# Checklist for adapting to incremental marking

# create_seed_grouping :: --> should no longer be needed
# mark_cells_vis_helper :: --> should be a simple function that returns a a copy of the visualization currently stored in mark_cells_editor_state
# plugin_mark_cells :: --> initialize the mark_cells_editor_state
# mark_cell :: --> Most important: perform incremental watershed
# delete_cell_mark :: --> Most important: reverse watershed over connected pixel islands
# plugin_mark_cells_save :: --> Should not require any change
# plugin_mark_cells_load :: --> Can either do the entire watershed at once, but a simpler solution is to just loop through the loaded marks
#                                   and "place it down" like a human would


"""
Plan of attack:

First, add initialization for the editor state in the plugin_mark_cells function
Next, adjust mark_cells_vis_helper to simply retrieve data from the editor state
Then, add functionality to mark cells
Then add functionality to delete cells (shift+click)
Save markings should not need to be changed
Finally, add functionality to refersh visuals after loading markings         
"""

def create_seed_grouping(marker_positions):
    
    marker_element_rowcol = np.transpose(np.nonzero(marker_element))

    H, W = active_image.shape[:2]
    seed_grouping = np.zeros((H, W),dtype=int) -1
    for idx, marker_position in enumerate(marker_positions):
        for dr, dc in marker_element_rowcol:
            try:
                seed_grouping[int(marker_position[1]+dr-marker_element.shape[0]//2),int(marker_position[0]+dc-marker_element.shape[1]//2)] = idx
            except IndexError as e:
                U.dprint(e)

    return seed_grouping

def draw_circle_on_image_around_point(img,x,y,color):
    
    offsx = marker_element.shape[1]//2
    offsy = marker_element.shape[0]//2
    
    for dx, dy in marker_element_xy:
        try:
            img[y-offsy+dy,x-offsx+dx,:] = color 
        except IndexError:
            pass
        
    return img

def restore_circle_on_image_around_point(img,x,y,pristine_img):
    
    offsx = marker_element.shape[1]//2
    offsy = marker_element.shape[0]//2
    
    for dx, dy in marker_element_xy:
        try:
            img[y-offsy+dy,x-offsx+dx,:] = pristine_img[y-offsy+dy,x-offsx+dx,:] 
        except IndexError:
            pass
        
    return img

def draw_circle_on_grouping_around_point(img,x,y,idx):
    
    offsx = marker_element.shape[1]//2
    offsy = marker_element.shape[0]//2
    
    for dx, dy in marker_element_xy:
        try:
            img[y-offsy+dy,x-offsx+dx] = idx
        except IndexError:
            pass
        
    return img
        
def mark_cells_vis_helper(return_numpy = False):
    
    # seed_grouping = create_seed_grouping(marker_positions)

    # seed_grouping_image = IP.grouping_to_color_image(seed_grouping)

    # dual_view_R = active_image.copy()

    # dual_view_R = IP.Image.binary_mix(dual_view_R,seed_grouping_image, seed_grouping!=-1)

    # # It appears that watershed purely on the mask may not be so wise. 
    # # The conservative cell mask often pickes up the 

    # LAB_image = cv2.cvtColor(active_image, cv2.COLOR_RGB2LAB)

    # L_channel = U.rescale_array(LAB_image[:,:,0])
    # A_channel = U.rescale_array(LAB_image[:,:,1])
    # B_channel = U.rescale_array(LAB_image[:,:,2])

    

    # watershed_channel = IP.greyscale_plot_to_color_image(L_channel)

    # seed_grouping_watershed = IP.Grouping.watershed(active_cell_mask, seed_grouping,watershed_channel)

    # seed_grouping_watershed_image = IP.grouping_to_color_image(seed_grouping_watershed)

    # seed_grouping_watershed_image = IP.Image.binary_mix(IP.greyscale_plot_to_color_image(active_cell_mask), seed_grouping_watershed_image, seed_grouping_watershed!=-1)

    # dual_view_L = seed_grouping_watershed_image

    # if return_numpy:
    #     return dual_view_L, dual_view_R

    # return json.dumps([image_to_json_compatible_obj(dual_view_L),image_to_json_compatible_obj(dual_view_R)]) # auxillary, main
    
    if return_numpy:
        return (mark_cells_editor_state.vis_annotated_cell_mask,mark_cells_editor_state.vis_annotated_image)
    
    return json.dumps([
        image_to_json_compatible_obj(mark_cells_editor_state.vis_annotated_cell_mask), image_to_json_compatible_obj(mark_cells_editor_state.vis_annotated_image)
    ])
    

    

def init_editor_state():
    
    # Initialize editor state
    mark_cells_editor_state.active_image = active_image
    mark_cells_editor_state.vis_annotated_image = np.copy(active_image)
    mark_cells_editor_state.active_cell_mask = active_cell_mask
    mark_cells_editor_state.vis_annotated_cell_mask = IP.greyscale_plot_to_color_image(active_cell_mask)
    # Use a connectivity of 8 to be as conservative as possible. Want to avoid a case where a perceptally adjacent pixel island is wrongly excluded from incremental waterhed
    mark_cells_editor_state.cluster_grouping = IP.BooleanImageOps.connected_components(active_cell_mask,8)
    mark_cells_editor_state.markers_grouping = np.zeros_like(mark_cells_editor_state.cluster_grouping).astype(int) - 1
    

@eel.expose
def plugin_mark_cells(clear_markers=True):
    
    global active_cell_mask
    global active_plugin
    global marker_positions
    
    if active_plugin == "mark_cells":
        plugin_console_log("Already started marking cells. To clear all markings, use file->open.")
        return

    if clear_markers:
        while len(marker_positions) >0:
            marker_positions.pop()

    active_plugin="mark_cells"
    active_cell_mask = get_rpe_nucleus_mask(active_image)
    
    init_editor_state()
    
    return mark_cells_vis_helper()

def incremental_vis_update_add_mark(x,y):
    
    current_cell_idx = len(marker_positions) - 1 
    
    current_color = IP.random_color_at_index(current_cell_idx)
    
    mark_cells_editor_state.vis_annotated_image = draw_circle_on_image_around_point(mark_cells_editor_state.vis_annotated_image,x,y,current_color)
    
    mark_cells_editor_state.markers_grouping = draw_circle_on_grouping_around_point(mark_cells_editor_state.markers_grouping,x,y,current_cell_idx)
    
    enclosing_cluster_id = mark_cells_editor_state.cluster_grouping[y,x] 
    
    enclosing_cluster = IP.BooleanImageOps.where2D(mark_cells_editor_state.cluster_grouping==enclosing_cluster_id)
    
    tlc,brc = IP.bounds_of_island(enclosing_cluster)
    
    cluster_slice = IP.corners_to_slice(tlc,brc)
    
    image_section = active_image[cluster_slice]
    
    cell_mask_section = active_cell_mask[cluster_slice]
    
    grouping_section = mark_cells_editor_state.markers_grouping[cluster_slice]
    
    watershed_grouping_section = perform_watershed(image_section,cell_mask_section,grouping_section)
    
    vis_image_section = IP.greyscale_plot_to_color_image(cell_mask_section)
    vis_image_section_2 = IP.grouping_to_color_image(watershed_grouping_section)
    vis_image_section[watershed_grouping_section!=-1] = vis_image_section_2[watershed_grouping_section!=-1]
    
    
    mark_cells_editor_state.vis_annotated_cell_mask[cluster_slice] = vis_image_section
    
   

def incremental_vis_update_delete_mark(x,y):
    
    mark_cells_editor_state.vis_annotated_image = restore_circle_on_image_around_point(mark_cells_editor_state.vis_annotated_image,x,y,active_image)
    
    mark_cells_editor_state.markers_grouping = draw_circle_on_grouping_around_point(mark_cells_editor_state.markers_grouping,x,y,-1)
    
    enclosing_cluster_id = mark_cells_editor_state.cluster_grouping[y,x] 
    
    enclosing_cluster = IP.BooleanImageOps.where2D(mark_cells_editor_state.cluster_grouping==enclosing_cluster_id)
    
    tlc,brc = IP.bounds_of_island(enclosing_cluster)
    
    cluster_slice = IP.corners_to_slice(tlc,brc)
    
    image_section = active_image[cluster_slice]
    
    cell_mask_section = active_cell_mask[cluster_slice]
    
    grouping_section = mark_cells_editor_state.markers_grouping[cluster_slice]
    
    watershed_grouping_section = perform_watershed(image_section,cell_mask_section,grouping_section)
    
    vis_image_section = IP.greyscale_plot_to_color_image(cell_mask_section)
    vis_image_section_2 = IP.grouping_to_color_image(watershed_grouping_section)
    vis_image_section[watershed_grouping_section!=-1] = vis_image_section_2[watershed_grouping_section!=-1]
    
    mark_cells_editor_state.vis_annotated_cell_mask[cluster_slice] = vis_image_section

@eel.expose
def mark_cell(x,y):

    global marker_positions

    x=int(x)
    y=int(y)
    # dual_view = np.hstack((IP.greyscale_plot_to_color_image(active_cell_mask),active_image))

    if x < 0 or x >= active_cell_mask.shape[1] or y < 0 or y >= active_cell_mask.shape[0]:
        return

    if not active_cell_mask[y,x]:
        return
    
    marker_positions.append((x,y))

    plugin_console_log(f"Num Cells: {len(marker_positions)}")

    print(f"Num Cells: {len(marker_positions)}")
    
    incremental_vis_update_add_mark(x,y)

    return mark_cells_vis_helper()

@eel.expose
def delete_cell_mark(x,y):

    global marker_positions

    x=int(x)
    y=int(y)
    
    if x < 0 or x >= active_cell_mask.shape[1] or y < 0 or y >= active_cell_mask.shape[0]:
        return

    color_cache = IP.get_color_cache()

    new_marker_positions = []
    new_color_cache = []

    delete_position = None
    
    for idx, pos in enumerate(marker_positions):
        if np.linalg.norm(np.array((x,y,),float)-np.array(pos,float)) > MARKER_DELETE_RADIUS:
            new_marker_positions.append(pos)
            new_color_cache.append(color_cache[idx])
        else:
            
            delete_position = pos

            IP.set_color_cache(new_color_cache)

            marker_positions = new_marker_positions

            plugin_console_log(f"Marker deleted at x={x}, y={y}.")

            plugin_console_log(f"Num Cells: {len(marker_positions)}")

            print(f"Num Cells: {len(marker_positions)}")
            
            incremental_vis_update_delete_mark(delete_position[0],delete_position[1])
            
    return mark_cells_vis_helper()

@eel.expose
def plugin_mark_cells_save():

    if active_plugin != "mark_cells":
        plugin_console_log("Nothing to save.")
        return

    if active_image_path is None:
        raise Exception("No path was specified. Has an image been loaded?")

    save_dir = active_image_path+".annotated/"

    U.init_folder(save_dir,clear=True)

    L,R = mark_cells_vis_helper(return_numpy=True)

    IP.ImageStrip(active_image,IP.greyscale_plot_to_color_image(active_cell_mask),L,R,max_cols = 1 if active_image.shape[1]/active_image.shape[0] >= VERTICAL_MIN_RATIO else None ).saveSelf(save_dir+"images.png")

    """save statistics to text file"""
    num_cells = len(marker_positions)
    with open(save_dir+"stats.txt","w") as fl:
        fl.write(f"Num cells: {num_cells}\n")
    

    """Save program state"""
    save_state = {}
    save_state["marker_positions"] = marker_positions
    save_state["active_image"] = active_image
    save_state["active_image_path"] = active_image_path
    save_state["active_sf"] = active_sf
    with open(save_dir+'save_state.markcells.pkl','wb') as fl:
        pickle.dump(save_state,fl)
    
@eel.expose
def plugin_mark_cells_reset():
    global marker_positions
    save_dir = active_image_path+".annotated/"
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    marker_positions = []
    init_editor_state()
    return mark_cells_vis_helper()
    
@eel.expose
def plugin_mark_cells_load():
    global marker_positions

    filepath = simplefilepicker.askopenfilename(wildcard="*.markcells.pkl")
    if filepath:
        with open(filepath,'rb') as fl:

            if active_plugin != "mark_cells":
                plugin_mark_cells(False)
                
            save_state = pickle.load(fl)
            marker_positions = save_state["marker_positions"]
            
            for marker_position in marker_positions:
                print(marker_position)
                incremental_vis_update_add_mark(*marker_position)
            
            return mark_cells_vis_helper()
    else:
        print("User has canceled load.")

""" --------------------------- """

@eel.expose
def read_messages():
    result = []
    while len(messages) > 0:
        result.append(messages.popleft())
    return json.dumps(result)

eel.start('index.html')