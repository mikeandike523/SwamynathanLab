"""Initialize program environment"""
import env_setup
env_setup.env_setup()

"""Built in imports"""
import math
import copy
import os
import argparse
import pickle
from collections import Counter
import functools
import shutil

"""3rd party imports"""
import numpy as np
import cv2
import scipy.spatial
import scipy.ndimage
import matplotlib.pyplot as plt
import PIL.Image
import pandas as pd
import dill

"""internal imports"""
import image_processing as IP
import utils as U

"""Program Constants"""
NUM_GROUPS = 30
#@TODO verify if downsampling by a factor of 1 does not change image, otherwise need to add additional if statement
DOWNSAMPLING = 1 
CANNY_LOW=25
CANNY_HIGH=240
SEPARATION_BLOCKSIZE = 94
RATIO_EPSILON = 1.0
DISCRIMINATOR_MASS_MAX_DEVIATIONS = 0.5
DISCRIMINATOR_VARIATION_MAX_DEVIATIONS = 0.5
CORRECTION_FACTOR_NUM_BINS = 32


"""Functions to apply neural network to seperate cells"""

# Add path for new project's library
import sys
sys.path.insert(0,"D:\\SwamynathanLab\\Retina Mapping Project")

#create a pre-trained evaulator
from core.deeplearning import CellDifferentiator

#Previously trained file was with SZ = 128 and nBaseFilters = 128

pretrained_path = "D:\\SwamynathanLab\\Retina Mapping Project\\output\\segmentation.h5"

differentiator = CellDifferentiator(128,128,pretrained_path)


def run_differentiator(pixels):
    retval =differentiator.apply_to_images(np.array([pixels]))[0]
    return retval

output_test_counter = 0

def get_neural_network_output(masked_image_white_background):

    global output_test_counter

    H,W = masked_image_white_background.shape[:2]

    windowable_input_image = IP.to_WindowableRGBImage(masked_image_white_background)

    windowable_output_image = IP.to_WindowableRGBImage(np.zeros_like(masked_image_white_background))

    locations = U.Geometry.get_window_locations_covering_image(
        W,H,128,128
    )

    for x,y in locations:
        input_window = windowable_input_image.read_window(x,y,128,128)
        output_window = run_differentiator(input_window)
        IP.imshow(output_window,f"NN_output_test/NN_output_{output_test_counter}_{x}_{y}")
        windowable_output_image.write_window(x,y,128,128,np.array(output_window,np.uint8))

    output_test_counter += 1

    return windowable_output_image.getImagePixels()

def predict_labels(masked_image_white_background):

    output_pixels = get_neural_network_output(masked_image_white_background)

    mask = np.mean(output_pixels.astype(float)/255,axis=2) > 0.5

    labels = IP.BooleanImageOps.connected_components(mask,4)

    return labels


"""Calculate the spine of the cell layer mask and related data"""
import get_initial_data
initial_data = get_initial_data.exports()

"""Obtain important information from initial_data variable"""
ds_mask = copy.deepcopy(initial_data.fs_mask)
ds_boundary = copy.deepcopy(initial_data.fs_boundary)
ds_spine = copy.deepcopy(initial_data.fs_spine)
ds_image = copy.deepcopy(initial_data.fs_image)
ds_H, ds_W = ds_image.shape[:2]
fs_mask_image = np.dstack((np.where(ds_mask,255,0).astype(np.uint8),)*3)
fs_mask_image = IP.scale_image_nearest(fs_mask_image,1/DOWNSAMPLING,1/DOWNSAMPLING)
ds_mask = fs_mask_image[:,:,0] > 0
ds_boundary = 1/DOWNSAMPLING * np.array(ds_boundary,float)
ds_spine = 1/DOWNSAMPLING * np.array(ds_spine,float) 
ds_image = IP.scale_image_nearest(ds_image,1/DOWNSAMPLING,1/DOWNSAMPLING)
ds_spine = IP.Curve.resample_by_segment_length(ds_spine,12.5,False)
fs_tangents, fs_normals = IP.Curve.curve_tangents_and_normals(ds_spine)

"""Calculate how a segments of the resampled spine are distributed among the groups"""
num_segments_per_group = math.floor(len(ds_spine)/NUM_GROUPS)
num_unaccounted = len(ds_spine) - NUM_GROUPS*num_segments_per_group
start_point = num_unaccounted // 2

"""Initialize filesystem and handle command line arguments"""
if not os.path.isdir("temp/group_mask_cache"):
    os.mkdir("temp/group_mask_cache")
parser = argparse.ArgumentParser()
parser.add_argument("--reset-state",action="store_true")
args = parser.parse_args()
if args.reset_state:
    U.init_folder("temp/group_mask_cache")
U.init_folder("debug")
U.init_folder("debug/group_masks_for_manual_counting")
U.init_folder("debug/discriminator_test")
U.init_folder("debug/NN_output_test/",clear=True)


"""Function to get a mask representing the portion of the desired cell layer corresponding to a particular group"""
def get_group_mask(group_number):
    if os.path.isfile(f"temp/group_mask_cache/{group_number}.pkl"):
        with open(f"temp/group_mask_cache/{group_number}.pkl","rb") as fl:
            return pickle.load(fl)
    start_divider = group_number * num_segments_per_group
    end_divider = (group_number + 1) * num_segments_per_group
    if end_divider >= len(ds_spine):
        end_divider = len(ds_spine) - 1
    nonzero_points = np.transpose(np.nonzero(ds_mask))
    group_mask = np.zeros_like(ds_mask)
    start_divider_pos = ds_spine[start_divider]
    end_divider_pos = ds_spine[end_divider]
    start_divider_tangent = fs_tangents[start_divider]
    end_divider_tangent = fs_tangents[end_divider]
    for nonzero_point in nonzero_points:
        y,x = nonzero_point
        start_dx = x - start_divider_pos[0]
        start_dy = y - start_divider_pos[1]
        end_dx = x - end_divider_pos[0]
        end_dy = y - end_divider_pos[1]
        t1 = U.scalar_project(start_divider_tangent,(start_dx,start_dy))
        t2 = U.scalar_project(end_divider_tangent,(end_dx,end_dy))
        if t1 > 0 and t2 < 0:
            group_mask[y,x] = True
    with open(f"temp/group_mask_cache/{group_number}.pkl","wb") as fl:
        pickle.dump(group_mask,fl)
    
    return group_mask


"""Additional helper functions"""
def isBlank(img):
    return functools.reduce(np.logical_and,[img[:,:,c]==0 for c in range(3)])
def overlay(under,over):
    blank_area = isBlank(under)
    return np.where(np.dstack((blank_area,)*3),over,under)

"""Calculate all of the group masks"""
group_masks = []
group_mask_visulization = np.zeros_like(ds_image)
group_mask_indices_plot = np.zeros(ds_image.shape[:2]) - 1
def lpad(string,length):
    while len(string) < length:
        string = " "+string
    return string

num_digits = len(f"{NUM_GROUPS}")
for i in range(NUM_GROUPS):
    print(f"Processing group {i}...")
    group_mask = get_group_mask(i)
    group_mask_visulization=overlay(group_mask_visulization,IP.grouping_to_color_image(np.where(group_mask,i,-1)))
    group_mask_indices_plot[group_mask] = i
    start_divider = i * num_segments_per_group
    end_divider = (i + 1) * num_segments_per_group 
    mid_divider =int(0.5*start_divider + 0.5*end_divider)
    mid_divider_normal = fs_normals[mid_divider]
    x,y = (np.array(ds_spine[mid_divider])+35*mid_divider_normal).astype(int)
    group_mask_visulization = overlay(IP.Draw.draw_annotation(group_mask_visulization,lpad(f"{i}",num_digits),x,y,7,(255,0,0),5,1.0,5),group_mask_visulization)
    group_masks.append(group_mask)
IP.imshowjpg(group_mask_visulization,"group mask visualization",90)

"""List of visualizations"""
vis_L = np.zeros((ds_H, ds_W),dtype=float)
vis_A = np.zeros((ds_H, ds_W),dtype=float)
vis_B = np.zeros((ds_H, ds_W),dtype=float)
vis_combined = np.zeros((ds_H, ds_W),dtype=float)
vis_conservative_cell_mask = np.zeros((ds_H,ds_W),dtype=float)
vis_H = np.zeros((ds_H, ds_W),dtype=float)
vis_L = np.zeros((ds_H, ds_W),dtype=float)
vis_S = np.zeros((ds_H, ds_W),dtype=float)
vis_combined_2 = np.zeros((ds_H, ds_W),dtype=float)
vis_separation_plot = np.zeros((ds_H, ds_W),dtype=float)
vis_separation_mask = np.zeros((ds_H, ds_W),dtype=float)
vis_watershed_grouping = np.zeros((ds_H,ds_W),dtype=int) - 1
vis_single_cells = np.zeros((ds_H,ds_W),dtype=int) - 1
vis_single_cell_edges = np.zeros((ds_H,ds_W),dtype=bool)
vis_neural_network_output = np.zeros((ds_H, ds_W,3),dtype = np.uint8)

"""Storage for cell metrics"""
mass_data = []
mass_ratio_data = []
uniformity_data = []
num_islands_after_filter = 0
island_count_data = []
group_island_data = [[] for _ in range(NUM_GROUPS)]
group_correction_factors = []
group_raw_cell_counts = []


for group_number, group_mask in enumerate(group_masks):
    print(f"Running pipeline for group {group_number}...")
    top_left_corner,bottom_right_corner = IP.BooleanImageOps.bounds(group_mask)
    group_slice = np.s_[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]
    group_mask = group_mask[group_slice]
    image_section = ds_image[group_slice]
    masked_image = IP.BooleanImageOps.apply_mask_to_image(group_mask,image_section)
    masked_image_white_background = image_section.copy()
    masked_image_white_background[~group_mask,:] = (255,255,255)

    counting_image = image_section.copy()
    counting_image[np.logical_not(np.dstack((group_mask,)*3))] = 255

    PIL.Image.fromarray(counting_image).save(f"debug/group_masks_for_manual_counting/{group_number}.png")
    masked_image_LAB = cv2.cvtColor(masked_image,cv2.COLOR_RGB2LAB)
    L= U.rescale_array(masked_image_LAB[:,:,0].astype(float))
    A= U.rescale_array(masked_image_LAB[:,:,1].astype(float))
    B= U.rescale_array(masked_image_LAB[:,:,2].astype(float))
    LAB_L= L.copy()
    vis_L[group_slice] = L
    vis_A[group_slice] = A
    vis_B[group_slice] = B
    combined = functools.reduce(np.multiply,[1.0-L])
    combined[~group_mask] = 0.0
    vis_combined[group_slice] = combined
    centers, labels2D, qimg = IP.unique_quantize(combined,2,True)
    conservative_cell_mask = labels2D == 1
    adaptive_cell_mask = IP.Channel.adaptive_threshold_by_mean(combined,21)
    conservative_cell_mask = np.logical_and(conservative_cell_mask,adaptive_cell_mask)
    element = IP.circular_structuring_element(1).astype(int)
    element[1,1] = 0
    ncounts = scipy.ndimage.convolve(conservative_cell_mask.astype(int),element)
    conservative_cell_mask[ncounts < 2] = False
    vis_conservative_cell_mask[group_slice] = np.where(conservative_cell_mask,1,0)
    masked_image_HLS = cv2.cvtColor(masked_image,cv2.COLOR_RGB2HLS_FULL)
    H = U.rescale_array(masked_image_HLS[:,:,0].astype(float))
    L = U.rescale_array(masked_image_HLS[:,:,1].astype(float))
    S = U.rescale_array(masked_image_HLS[:,:,2].astype(float))
    H[~conservative_cell_mask] = 0.0
    L[~conservative_cell_mask] = 0.0
    S[~conservative_cell_mask] = 0.0
    vis_H[group_slice] = H
    vis_L[group_slice] = L
    vis_S[group_slice] = S
    combined_2 = functools.reduce(np.multiply,[1.0-L,S])
    vis_combined_2[group_slice] = combined_2
    dst = IP.BooleanImageOps.distance_transform_L1(conservative_cell_mask)
    grouping = IP.BooleanImageOps.connected_components(conservative_cell_mask,4)
    adjusted_dst = np.zeros_like(dst)
    islandsDict = IP.Grouping.getIslands(grouping)
    for label, island in islandsDict.items():
        max_distance = np.max(dst[grouping==label])
        for x,y in island:
            adjusted_dst[y,x] = dst[y,x] / max_distance
    separation_plot = U.rescale_array(np.multiply(combined_2,adjusted_dst))
    vis_separation_plot[group_slice] = separation_plot
    centers, labels2D, qimg = IP.unique_quantize(separation_plot,2,ignore_zero = True)
    separation_mask = labels2D == 1
    vis_separation_mask[group_slice] = np.where(separation_mask,1,0)


    # seed_grouping = IP.BooleanImageOps.connected_components(separation_mask,4)

    vis_neural_network_output[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0],:] = get_neural_network_output(masked_image_white_background) 
    seed_grouping = predict_labels(masked_image_white_background)


    watershed_hint = seed_grouping + 2
    watershed_hint[np.logical_and(conservative_cell_mask,np.logical_not(separation_mask))] = 0
    watershed_image =IP.greyscale_plot_to_color_image(conservative_cell_mask)
    markers = cv2.watershed(watershed_image,watershed_hint)
    markers[markers==1] = -1
    markers[markers>=2] -=2
    watershed_grouping = markers
    vis_watershed_grouping[group_slice] = watershed_grouping
    single_cells = np.zeros_like(watershed_grouping) -1
    islandsDict = IP.Grouping.getIslands(watershed_grouping)
    num_islands = len(islandsDict)
    total_pixels = 0

    
    discriminator_mass_data = {}
    discriminator_variation_data = {}

    discriminator_all_cell_vis = np.zeros_like(watershed_grouping) - 1
    discriminator_single_cell_vis = np.zeros_like(watershed_grouping) - 1
    discriminator_masked_image_vis = masked_image.copy()


    for idx, (label, island) in enumerate(islandsDict.items()):
        print(f"Island {idx+1} of {num_islands}...")        
        try:
            island_mask = IP.island_to_mask(island)
            convex_hull = np.squeeze(cv2.convexHull(island,returnPoints=True))
            convex_hull_mask = IP.Curve.curve_to_mask(convex_hull)
            ratio = np.count_nonzero(island_mask) / np.count_nonzero(convex_hull_mask)
            island_boundary = IP.BooleanImageOps.subject_boundary(island_mask,3)
            if island_boundary is None or len(island_boundary) ==0:
                print("Could not get island boundary")
                continue
            island_center = np.flip(np.mean(np.transpose(np.nonzero(island_mask)),axis=0))
            distances = []
            for pt in island_boundary:
                distances.append(np.linalg.norm(pt-island_center))
            print(len(distances))
            variation = np.var(distances)/(np.mean(distances)**2)
            total_pixels += len(island)
            group_island_data[group_number].append(
                {
                    "mass":len(island),
                    "variation":variation
                }
            )
            for x,y in island:
                single_cells[y,x] = label
            num_islands_after_filter += 1
            uniformity_data.append(variation)
            mass_ratio_data.append(ratio)
            mass_data.append(len(island))
            discriminator_mass_data[label]=len(island)
            discriminator_variation_data[label]=variation
        except Exception as e:
            print(e)

    mean_mass = np.mean(list(discriminator_mass_data.values()))
    mean_variation = np.mean(list(discriminator_variation_data.values()))
    mass_std = np.std(list(discriminator_mass_data.values()))
    variation_std = np.std(list(discriminator_variation_data.values()))

    unfiltered_islands = []
    filtered_islands = []

    # Filter pixel islands by mass and variation
    for label, island in islandsDict.items():
        
        if label not in discriminator_mass_data or label not in discriminator_variation_data:
            continue

        mass = discriminator_mass_data[label]
        variation = discriminator_variation_data[label]

        mass_diff = abs(mass-mean_mass)
        variation_diff = abs(variation-mean_variation)

        unfiltered_islands.append((label,island))

        if mass_diff <= mass_std * DISCRIMINATOR_MASS_MAX_DEVIATIONS and variation_diff <= variation_std * DISCRIMINATOR_VARIATION_MAX_DEVIATIONS:
            filtered_islands.append((label,island))

    # Create visualizations for the single-cell discriminator
    for label, island in unfiltered_islands:
        for x,y in island:
            discriminator_all_cell_vis[y,x] = label
    for label, island in filtered_islands:
        for x,y in island:
            discriminator_single_cell_vis[y,x] = label
    PIL.Image.fromarray(IP.ImageStrip(discriminator_masked_image_vis,
        IP.grouping_to_color_image(discriminator_all_cell_vis),
        IP.grouping_to_color_image(discriminator_single_cell_vis)
    ).getImagePixels()).save(f"debug/discriminator_test/group_{group_number}.png")

    # determine correction factor
    # counts, bin_edges = np.histogram([len(filtered_islands[1]) for filtered_island in filtered_islands])
    # bin_centers = U.bin_centers(bin_edges)
    # weights = counts.astype(float) / np.sum(counts.astype(float))
    # correction_factor = np.sum(np.multiply(bin_centers, weights))
    correction_factor = np.mean([len(filtered_island[1]) for filtered_island in filtered_islands])
    group_correction_factors.append(correction_factor)
    group_raw_cell_counts.append(len(unfiltered_islands))

    island_count_data.append(total_pixels/correction_factor)
    vis_single_cells[group_slice] = single_cells
    total = scipy.ndimage.convolve(single_cells,np.array([[0,1,0],[1,1,1],[0,1,0]]),mode='constant',cval=-1)
    single_cell_edges = total!=5*single_cells
    vis_single_cell_edges[group_slice] = single_cell_edges


"""Output relavant visualizations"""
U.dprint("Outputting relevant visualizations...")
IP.imshow(IP.greyscale_plot_to_color_image(vis_conservative_cell_mask),"Conservative Cell Mask")
IP.imshow(IP.grouping_to_color_image(vis_single_cells),"Single Cells")
edge_vis_image=ds_image.copy()
edge_vis_image[vis_single_cell_edges,...] = (0,255,0)
IP.imshow(edge_vis_image,"Single Cell Visualization")
IP.imshow(vis_neural_network_output, "Nueral network output")


"""Create images for manual verification"""
U.dprint("Creating images for manual verification...")
U.init_folder("debug/group_masks_for_manual_counting_shuffled")
group_numbers = np.array(list(range(NUM_GROUPS)))
np.random.seed(1999)
np.random.shuffle(group_numbers)
mapping = {idx:group_numbers[idx] for idx in range(len(group_numbers))}
for k, v in mapping.items():
    shutil.copy(f"debug/group_masks_for_manual_counting/{k}.png",f"debug/group_masks_for_manual_counting_shuffled/{v}.png")
print(mapping)


"""Print numerical measurements"""
print(f"num islands after filter: {num_islands_after_filter}")


"""Create plots for relevant statistics"""
plt.clf()
plt.bar(list(range(NUM_GROUPS)),island_count_data)
plt.title("Cell Count Along Retina")
plt.xlabel("Group Number")
plt.ylabel("Cell Count")
plt.savefig("debug/group_cell_counts.jpg")

"""Save relevant statistics to text file"""
with open("debug/group_cell_counts.txt", "w") as fl:
    for group_number in range(NUM_GROUPS):
        fl.write(f"Section {group_number}, Raw Cell Count{group_raw_cell_counts[group_number]}, Correction Factor {group_correction_factors[group_number]}, Cell Count {island_count_data[group_number]} \n")

"""Create and print dataframes for the mass and variation in the islands for each group"""
U.dprint("Assembling dataframes...")
frames = []
for idx, record_list in enumerate(group_island_data):
    print(f"{idx+1} of {len(group_island_data)}...")
    frames.append(pd.DataFrame.from_records(record_list))    

# """Output relevant plots for each group"""
# U.dprint("Creating plots for each group...")
# U.init_folder("debug/plots", clear=True)
# for group_number, df in enumerate(frames):

#     mass = df["mass"].values
#     variation= df["variation"].values
    
#     plt.figure()

#     plt.subplot(131)
#     plt.title("Mass Histogram")
#     counts, bin_edges = np.histogram(mass, bins = 32)
#     plt.stairs(counts, bin_edges)
#     plt.xlabel("mass")
#     plt.ylabel("frequency")

#     plt.subplot(132)
#     plt.title("Variation Histogram")
#     counts, bin_edges = np.histogram(variation, bins = 32)
#     plt.stairs(counts, bin_edges)
#     plt.xlabel("variation")
#     plt.ylabel("frequency")

#     plt.subplot(133)
#     plt.title("Mass-Variation Plot")
#     counts, bin_edges = np.histogram(variation, bins = 32)
#     plt.plot(mass,variation,'.')
#     plt.xlabel("mass")
#     plt.ylabel("variation")

#     plt.savefig(f"debug/plots/group_metrics_{group_number}.jpg")

"""Visualize the counts directly on the image"""

print("Creating group cell counts visualization...")

panorama_image = ds_image.copy()
masked_panorama_image = IP.BooleanImageOps.apply_mask_to_image(ds_mask,panorama_image)
group_cell_counts = island_count_data
print(island_count_data)
group_counts_visualization = masked_panorama_image.copy()

num_digits = len(f"{int(np.max(group_cell_counts))}")

print(num_digits)

for i in range(NUM_GROUPS):
    print(f"Processing group {i}...")
    group_mask = group_masks[i]
    start_divider = i * num_segments_per_group
    end_divider = (i + 1) * num_segments_per_group 
    mid_divider =int(0.5*start_divider + 0.5*end_divider)
    mid_divider_normal = fs_normals[mid_divider]
    x,y = (np.array(ds_spine[mid_divider])+30*mid_divider_normal).astype(int)
    print(f"{int(group_cell_counts[i])}")
    group_counts_visualization = overlay(IP.Draw.draw_annotation(group_counts_visualization,lpad(f"{int(group_cell_counts[i])}",num_digits),x,y,4,(255,0,0),2,1.1,2),group_counts_visualization)

edge_mask = IP.Grouping.get_edge_mask_from_grouping(group_mask_indices_plot,4)

group_counts_visualization = IP.BooleanImageOps.apply_mask_to_image(~edge_mask,group_counts_visualization,(255,0,0))

IP.imshowjpg(group_counts_visualization,"group cell counts visualization",95)