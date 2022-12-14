import env_setup

env_setup.env_setup()

import math
import copy
import os
import argparse
import pickle
from collections import Counter
import functools

import numpy as np
import cv2
import scipy.spatial
import scipy.ndimage
import matplotlib.pyplot as plt

import image_processing as IP
import utils as U

NUM_GROUPS = 15
#@TODO verify if downsampling by a factor of 1 does not change image, otherwise need to add additional if statement
DOWNSAMPLING = 1 
CANNY_LOW=25
CANNY_HIGH=240
SEPARATION_BLOCKSIZE = 94
RATIO_EPSILON = 1.0

import get_initial_data

initial_data = get_initial_data.exports()

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

num_segments_per_group = math.floor(len(ds_spine)/NUM_GROUPS)

num_unaccounted = len(ds_spine) - NUM_GROUPS*num_segments_per_group

start_point = num_unaccounted // 2

def scalar_project(base,target):
    return np.dot(target,base) / np.linalg.norm(base)

if not os.path.isdir("temp/group_mask_cache"):
    os.mkdir("temp/group_mask_cache")

parser = argparse.ArgumentParser()
parser.add_argument("--reset-state",action="store_true")
args = parser.parse_args()

if args.reset_state:
    U.init_folder("temp/group_mask_cache")

U.init_folder("debug")

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
        t1 = scalar_project(start_divider_tangent,(start_dx,start_dy))
        t2 = scalar_project(end_divider_tangent,(end_dx,end_dy))
        if t1 > 0 and t2 < 0:
            group_mask[y,x] = True

    with open(f"temp/group_mask_cache/{group_number}.pkl","wb") as fl:
        pickle.dump(group_mask,fl)
    
    return group_mask

group_masks = []

group_mask_visulization = np.zeros_like(ds_mask).astype(int) - 1
for i in range(NUM_GROUPS):
    print(f"Processing group {i}...")
    group_mask = get_group_mask(i)
    group_mask_visulization[group_mask] = i
    group_masks.append(group_mask)

IP.imshowjpg(IP.grouping_to_color_image(group_mask_visulization),"group mask visualization",90)

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

mass_data = []

mass_ratio_data = []

uniformity_data = []

num_islands_after_filter = 0

for group_number, group_mask in enumerate(group_masks):

    print(f"Running pipeline for group {group_number}...")

    top_left_corner,bottom_right_corner = IP.BooleanImageOps.bounds(group_mask)

    group_slice = np.s_[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]

    group_mask = group_mask[group_slice]

    image_section = ds_image[group_slice]

    masked_image = IP.BooleanImageOps.apply_mask_to_image(group_mask,image_section)

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

    seed_grouping = IP.BooleanImageOps.connected_components(separation_mask,4)

    watershed_hint = seed_grouping + 2

    watershed_hint[np.logical_and(conservative_cell_mask,np.logical_not(separation_mask))] = 0

    # watershed_image =IP.greyscale_plot_to_color_image(conservative_cell_mask)

    watershed_image = IP.greyscale_plot_to_color_image(np.where(group_mask,(1-B),0))

    markers = cv2.watershed(watershed_image,watershed_hint)

    markers[markers==1] = -1

    markers[markers>=2] -=2

    watershed_grouping = markers

    vis_watershed_grouping[group_slice] = watershed_grouping

    """Single-cell discriminator experiment"""

    single_cells = np.zeros_like(watershed_grouping) -1

    islandsDict = IP.Grouping.getIslands(watershed_grouping)

    num_islands = len(islandsDict)

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


            if variation <= 0.10 and len(island) >=25 and len(island) <= 200:
            #if True: #disable filtering
                for x,y in island:
                    single_cells[y,x] = label
                num_islands_after_filter += 1
                uniformity_data.append(variation)
                mass_ratio_data.append(ratio)
                mass_data.append(len(island))

            else:
                print(f"Island filtered out.")
        except Exception as e:
            print(e)

    vis_single_cells[group_slice] = single_cells

    total = scipy.ndimage.convolve(single_cells,np.array([[0,1,0],[1,1,1],[0,1,0]]),mode='constant',cval=-1)

    single_cell_edges = total!=5*single_cells

    vis_single_cell_edges[group_slice] = single_cell_edges

IP.imshow(IP.greyscale_plot_to_color_image(vis_conservative_cell_mask),"Conservative Cell Mask")
IP.imshow(IP.grouping_to_color_image(vis_single_cells),"Single Cells")
edge_vis_image=ds_image.copy()
edge_vis_image[vis_single_cell_edges,...] = (0,255,0)
IP.imshow(edge_vis_image,"Single Cell Visualization")

print(f"num islands after filter: {num_islands_after_filter}")

plt.clf()
plt.plot(mass_data, uniformity_data,'b.')
plt.title("Mass vs Variation After Filter")
plt.xlabel("Mass")
plt.ylabel("Variation")
plt.show()

counts, bin_edges = np.histogram(uniformity_data,bins=64)

plt.figure()
plt.stairs(counts,bin_edges)
plt.title("Variation Histogram After Filter")
plt.xlabel("Variation")
plt.ylabel("Frequency")
plt.show()

counts, bin_edges = np.histogram(mass_data, bins=64)

plt.figure()

plt.stairs(counts,bin_edges)

bin_centers = U.bin_centers(bin_edges)

print(bin_centers)

weighted_sum = np.sum(np.multiply(counts/np.sum(counts), bin_centers))

print(counts/np.sum(counts))

print(weighted_sum)

plt.plot([weighted_sum,weighted_sum],[np.min(counts),np.max(counts)],'r-')

plt.title("Mass Histogram After Filter")
plt.xlabel("Mass")
plt.ylabel("Frequency")
plt.show()