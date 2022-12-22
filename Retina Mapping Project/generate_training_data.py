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
MAX_IMAGES = float("inf")

SZ = 64 # Technically, it doesn't have to, but it should match the SZ used to train the network 

# Path to manually annotated data 
manually_annotated_data_path = normalize_path("G:\\My Drive\\Swamynathan Lab Image Processing\\Images for Manual Annotation 11_30_22\\dataset")

# List the images that have been manually marked. If an image has not been fully marked, this will be accounted for later in the module
manual_annotation_folders = [fn for fn in os.listdir(manually_annotated_data_path) if fn.endswith('.annotated')]

# Initialize output folder
init_folder("output/training_pairs",clear=True)

# Copied from the manual annotation project
# D:\SwamynathanLab\ImageJCloneWithPython\main.py

def create_seed_grouping(marker_positions,W,H,marker_radius):
    
    MARKER_RADIUS = marker_radius
    marker_element = IP.circular_structuring_element(MARKER_RADIUS,bool)

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
        
        marker_radius = saved_data["marker_radius"]

        H, W = image.shape[:2]

        marker_positions = saved_data["marker_positions"]
    
        # Create the markers based of the marker positions, using the same procedure as the manual annotation software

        seed_grouping = create_seed_grouping(marker_positions, W, H, marker_radius)

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
        # pair_output = IP.greyscale_plot_to_color_image(seed_grouping_watershed!=-1)
        pair_output = IP.greyscale_plot_to_color_image(seed_grouping!=-1)

        pair_input_windowable = IP.to_WindowableRGBImage(pair_input)
        pair_output_windowable = IP.to_WindowableRGBImage(pair_output)

        # Next, want to go through each island and limit the training image to a certain region around where the pixels are nonzero
        # islandsDict = IP.Grouping.getIslands(seed_grouping_watershed)

        # for label, island in islandsDict.items():

        #     min_x = np.min(island[:,0])
        #     max_x = np.max(island[:,0])        
        #     min_y = np.min(island[:,1])
        #     max_y = np.max(island[:,1])

        #     min_x -= PAD_PIXELS
        #     max_x += PAD_PIXELS
        #     min_y -= PAD_PIXELS
        #     max_y += PAD_PIXELS

        #     window_W = max_x - min_x + 1
        #     window_H = max_y - min_y + 1
            
        #     island_pair_input = pair_input_windowable.read_window(min_x,min_y,window_W,window_H)
        #     island_pair_output = pair_output_windowable.read_window(min_x,min_y,window_W,window_H)

        #     training_pair = np.hstack((island_pair_input,island_pair_output))

        #     id = uuid_in("output/training_pairs")

        #     PIL.Image.fromarray(training_pair).save(f"output/training_pairs/{id}.png")

        #     num_images +=1 

        #     print(f"Generated {num_images}/{MAX_IMAGES}.")

        #     if num_images == MAX_IMAGES:
        #         return

        


        # No need to create a single-island differentiator, if nearly all of the islands in the image are marked, the nueral network can just learn from sections of the image

        #However, will exclude images that are all one color. The application of the trained network should do the same



        H, W = image.shape[:2]

        num_x_positions = int(np.ceil(W / SZ))

        num_y_positions = int(np.ceil(H / SZ))

        x_positions = [int(SZ * i) for i in range(num_x_positions)]
        y_positions = [int(SZ * i) for i in range(num_y_positions)]

        BLACK = np.array([0,0,0],dtype=np.uint8)
        WHITE = np.array([255,255,255],dtype=np.uint8)

        for x in x_positions:
            for y in y_positions:
                window_pixels_input = pair_input_windowable.read_window_with_default(x,y,SZ,SZ,WHITE)

                if np.all(window_pixels_input==np.mean(window_pixels_input)):
                    continue

                window_pixels_output = pair_output_windowable.read_window_with_default(x,y,SZ,SZ,BLACK)
                fn = uuid_in("output/training_pairs") + ".png" 
                IP.ImageStrip(window_pixels_input,window_pixels_output,max_cols=2).saveSelf(f"output/training_pairs/"+fn)

                num_images += 1

                print(f"Created training image {num_images}.")

main()

# Verify the correct number of images were generated

print(f"Counting generated images...")
print(f"{len([fn for fn in os.listdir('output/training_pairs') if fn.endswith('.png')])}")

        

        
