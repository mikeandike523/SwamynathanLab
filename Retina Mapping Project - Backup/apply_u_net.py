"""! @brief A module to extract markings data from manually annotated RPE images and generate training data for neural networks."""

from core import setup
setup.setup()

import os
import pickle

import numpy as np
import cv2
import PIL.Image

import utils as U
import image_processing as IP

from core.path import join_paths, normalize_path, init_folder, uuid_in

from get_conservative_cell_mask import get_conservative_cell_mask


# Application Parameters
PAD_PIXELS = 5
MAX_IMAGES = 5000

# Path to manually annotated data 
manually_annotated_data_path = normalize_path("none yet... todo.")

# List the images that have been manually marked. If an image has not been fully marked, this will be accounted for later in the module
manual_annotation_folders = [fn for fn in os.listdir(manually_annotated_data_path) if fn.endswith('.annotated')]

# Initialize output folder
init_folder("output/apply_u_net/results",clear=True)

# Copied from the manual annotation project
# D:\SwamynathanLab\ImageJCloneWithPython\main.py

MARKER_RADIUS = 3

marker_element = IP.circular_structuring_element(MARKER_RADIUS,bool)

def create_seed_grouping(marker_positions,W,H):
    
    marker_element_rowcol = np.transpose(np.nonzero(marker_element))

    seed_grouping = np.zeros((H, W),dtype=int) -1
    for idx, marker_position in enumerate(marker_positions):
        for dr, dc in marker_element_rowcol:
            try:
                seed_grouping[int(marker_position[1]+dr-marker_element.shape[0]//2),int(marker_position[0]+dc-marker_element.shape[1]//2)] = idx
            except IndexError as e:
                U.dprint(e)

    return seed_grouping

# No need to "un-shuffle" the data. However, in the future, it will be useful to be able to associate the manually annotated sections withh the full panorama

np.random.shuffle(manual_annotation_folders)

def main():

    num_images = 0

    for idx, fn in enumerate(manual_annotation_folders):

        print(f"Preparing annotation image {idx+1} of {len(manual_annotation_folders)}...")

        with open(join_paths(manually_annotated_data_path, fn,"save_state.markcells.pkl"), 'rb') as fl:
            saved_data = pickle.load(fl)

        image = saved_data["active_image"]

        H, W = image.shape[:2]

        marker_positions = saved_data["marker_positions"]
    
        # Create the markers based of the marker positions, using the same procedure as the manual annotation software

        seed_grouping = create_seed_grouping(marker_positions, W, H)

        # Get the conservative cell mask from the image --> Same mask generated during manual annotation

        conservative_cell_mask = get_conservative_cell_mask(image)

        # Get watershed labels

        LAB_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        L_channel = U.rescale_array(LAB_image[:,:,0])
        A_channel = U.rescale_array(LAB_image[:,:,1])
        B_channel = U.rescale_array(LAB_image[:,:,2])

        watershed_channel = IP.greyscale_plot_to_color_image(L_channel)

        seed_grouping_watershed = IP.Grouping.watershed(conservative_cell_mask, seed_grouping,watershed_channel) 

        # Create training pair at full image scale

        # This is dependent on the neural network design

        # Current U-net design takes in a single-channel image, so it should be changed to take in RGB

        # For simplicity, can also output RGB

        pair_input = image
        pair_output = IP.greyscale_plot_to_color_image(seed_grouping_watershed!=-1)

        pair_input_windowable = IP.to_WindowableRGBImage(pair_input)
        pair_output_windowable = IP.to_WindowableRGBImage(pair_output)

        # Next, want to go through each island and limit the training image to a certain region around where the pixels are nonzero
        islandsDict = IP.Grouping.getIslands(seed_grouping_watershed)

        


        

        
