from core import setup
setup.setup()

import PIL.Image
import numpy as np
import cv2
from scipy.spatial import KDTree
import scipy.ndimage
from core.checkpoint import CheckpointManager
from core.database import Database, MapProxy
from core.debugging import imshow
from core.types import fluiddict
from core.progress import Progress
from core.io import imload, imsave
from core.path import init_folder
from core.debugging import SequentialNames
import utils as U
import image_processing as IP
from simplefilepicker import askopenfilename
import functools
import os

# Application Parameters

QUANTIZATION_PASS_1 = fluiddict(
    MIN_K = 3,
    MAX_K = 10,
    MIN_DIFF = 0.10,
    NUM_TOP_LEVELS = 2
)

QUANTIZATION_PASS_2 = fluiddict(
    MIN_K = 2,
    MAX_K = 5,
    MIN_DIFF = 0.20,
    NUM_TOP_LEVELS = 1
)

MORPH_CLEAN = fluiddict(
    R=3,
    ITERS=1,
    REPEATS=2
)

CLEAN_WITH_ALPHASHAPES = fluiddict(
    DS = 2,
    ALPHA = 0.075,
    MIN_PIXELS = 10_000
)

MORPH_CLEAN_2 = fluiddict(
    R=3,
    ITERS=1,
    REPEATS=2
)

seq_cell_mask = SequentialNames("cell_mask")

cm = CheckpointManager("main.2.py",True)

def preamble(selection):
    if selection in ["r","f"] or selection == "0":
        init_folder("debug",clear=True)
        
@cm.checkpoint
def pick_file(db: Database):
    
    panorama_path = askopenfilename()
    
    if panorama_path is None:
        print("No panorama selected. Quitting...")
        return cm.stop_here()
        
    panorama_image = imload(panorama_path)
    
    if panorama_image.shape[2] != 3:
        print(f"Image has more than 3 channels. Quitting...")
        return cm.stop_here()
        
    db.set_variable("panorama_image", panorama_image)
    
    H, W = panorama_image.shape[:2]
    
    db.set_variable("W",W)
    db.set_variable("H",H) 
    
    db.set_variable("panorama_path",panorama_path)
    
    imshow(panorama_image, "Panorama Image") 
    
@cm.checkpoint
def convert_to_lab_channels(db: Database):
    
    panorama_image = db.get_variable("panorama_image")
    
    panorama_image_LAB = cv2.cvtColor(panorama_image,cv2.COLOR_RGB2LAB)
    
    L, A, B = [panorama_image_LAB[:,:,c] for c in range(3)]
    
    L = U.rescale_array(L.astype(float))
    A = U.rescale_array(A.astype(float))
    B = U.rescale_array(B.astype(float))
    
    db.set_variable("L_channel",L)
    db.set_variable("A_channel",A)
    db.set_variable("B_channel",B)
    
    imshow(IP.greyscale_plot_to_color_image(L),"L Channel")
    imshow(IP.greyscale_plot_to_color_image(A),"A Channel")
    imshow(IP.greyscale_plot_to_color_image(B), "B Channel")
    
    combined = U.rescale_array(functools.reduce(np.multiply,[1.0-L,1.0-B,A]))
    
    db.set_variable("combined", combined)
    
    imshow(IP.greyscale_plot_to_color_image(combined),"Combined")

def get_mask_from_quantization(arr, min_k, max_k, min_diff, num_top_levels, ignore_zero=False):
    
    chosen_k = min_k
    
    for k in range(min_k, max_k + 1):
        
        centers, labels2D, qimg = IP.unique_quantize(arr,k,ignore_zero)
        
        assert not np.any(np.diff(centers) < 0)
        
        if np.any(np.diff(centers) < min_diff):
            chosen_k = k-1
            break
        
        print(str(k)+" | "+" ".join(map(lambda num: "{:.3f}".format(num),centers)) + " | " + " ".join(map(lambda num: "{:.3f}".format(num),np.diff(centers))))
        
    centers, labels2D, qimg = IP.unique_quantize(arr,chosen_k,ignore_zero)
    
    mask = labels2D >= (np.max(labels2D) + 1 - num_top_levels)
    
    return mask
    
@cm.checkpoint
def quantization_pass_1(db: Database):
    
    cell_mask_name = seq_cell_mask.next()
    
    combined = db.get_variable('combined')
    
    settings = QUANTIZATION_PASS_1
    
    mask = get_mask_from_quantization(combined,settings.MIN_K, settings.MAX_K, settings.MIN_DIFF, settings.NUM_TOP_LEVELS,False)
    
    db.set_variable(cell_mask_name,mask)
    
    imshow(IP.greyscale_plot_to_color_image(mask), cell_mask_name)
    
@cm.checkpoint
def quantization_pass_2(db: Database):
    
    cell_mask_name = seq_cell_mask.next()
    
    channel = 1.0-db.get_variable('L_channel')
    
    cell_mask = db.get_variable(seq_cell_mask.previous())
    
    channel[~cell_mask] = 0.0
    
    settings = QUANTIZATION_PASS_2
    
    mask = get_mask_from_quantization(channel,settings.MIN_K, settings.MAX_K, settings.MIN_DIFF, settings.NUM_TOP_LEVELS,True)
    
    db.set_variable(cell_mask_name,mask)
    
    imshow(IP.greyscale_plot_to_color_image(mask), cell_mask_name)

@cm.checkpoint
def morph_clean_mask(db: Database):
    
    cell_mask_name = seq_cell_mask.next()
    
    cell_mask = db.get_variable(seq_cell_mask.previous())
    cell_mask = IP.BooleanImageOps.openWithKernel(cell_mask,MORPH_CLEAN.R,MORPH_CLEAN.ITERS,MORPH_CLEAN.REPEATS)
    db.set_variable(cell_mask_name,cell_mask)
    imshow(cell_mask,cell_mask_name)

@cm.checkpoint
def clean_with_alphashapes(db: Database):
    
    cell_mask_name = seq_cell_mask.next()
    
    pr = Progress("clean_with_alphashapes")
    
    cell_mask = db.get_variable(seq_cell_mask.previous())
    
    H, W = db.get_variable('H'), db.get_variable('W')
    
    cell_mask = IP.BooleanImageOps.scale_by(cell_mask,1/CLEAN_WITH_ALPHASHAPES.DS)
    
    with pr.track("get nonzero locations"):
        pixel_locations_xy = np.flip(np.transpose(np.nonzero(cell_mask)),axis=1)

    with pr.track("get alphashapes"):
        polys = IP.run_alphashape_and_get_polygons(pixel_locations_xy,CLEAN_WITH_ALPHASHAPES.ALPHA)

    with pr.track("write database"):

        db.set_variable("island_polys",polys) #@TODO: double check that variables are indirect by default

    cell_mask = np.zeros_like(cell_mask)

    HH, WW = cell_mask.shape

    for poly in polys:

        curve = np.squeeze(poly.boundary.coords).astype(int)

        with pr.track("creating mask for alphashape polyon"):

            mask = IP.Curve.curve_to_mask(curve,WW,HH)

        cell_mask = np.logical_or(cell_mask,mask)

    cell_mask = IP.BooleanImageOps.resize_to(cell_mask,W,H)

    cell_mask = IP.BooleanImageOps.fast_filter_small_islands_and_get_grouping(cell_mask,4,CLEAN_WITH_ALPHASHAPES.MIN_PIXELS) != -1
    
    imshow(IP.greyscale_plot_to_color_image(cell_mask),cell_mask_name)
    
cm.menu(preamble)