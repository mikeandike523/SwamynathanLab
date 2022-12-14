#Attributions
# U-net architecture design adapted from https://blog.paperspace.com/unet-architecture-image-segmentation/
#       see u_net.py
# BatchProvider class concept adapted from https://blog.paperspace.com/unet-architecture-image-segmentation/
# Custom Callback design tutorial: https://www.tensorflow.org/guide/keras/custom_callback


#import necessary built in packages
import pickle
import math

# Import Necessary 3rd party Packages
import tensorflow as tf
import numpy as np
import random
import cv2
from scipy.spatial import KDTree
import PIL
import scipy.ndimage

PIL.Image.MAX_IMAGE_PIXELS = None

# Import custom scripts
import image_processing as IP
import u_net as UNET
import utils as U

#seed rng
random.seed()
np.random.seed()

SZ = 128

model = UNET.UNetModel(SZ,SZ)

"""Dont actually have to compile model when loading existing weights"""
#model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE), loss="mean_squared_logarithmic_error",metrics=["mean_squared_logarithmic_error","mean_squared_error"])

U.init_folder("debug/panorama_snippet_results")

model.load_weights("temp/segmentation.h5")

retina_map = np.asarray(PIL.Image.open("../assets/golden_retina_map.png"))
roi_mask_image = np.asarray(PIL.Image.open("../assets/dev/current_best_mask.png"))

roi_mask = roi_mask_image[:,:,0] > 0

masked_image = IP.BooleanImageOps.apply_mask_to_image(roi_mask,retina_map)

IP.imshow(masked_image,"masked_retina_map")

R, G, B = [masked_image[:,:,c].astype(float)/255 for c in range(3)]

RChannel = IP.Channel.from_array(R)
GChannel = IP.Channel.from_array(G)
BChannel = IP.Channel.from_array(B)

def get_window(x,y,w,h):
    """@Todo implement a version of nbChannel that takes uint8 as data,
        so will not corrupt data due to int->float->int conversion"""
    Rwin = (255*RChannel.read_window(x,y,w,h)).astype(np.uint8)
    Gwin = (255*GChannel.read_window(x,y,w,h)).astype(np.uint8)
    Bwin = (255*BChannel.read_window(x,y,w,h)).astype(np.uint8)
    return np.dstack((Rwin,Gwin,Bwin))

H, W = masked_image.shape[:2]

num_x_pos = math.ceil(W/SZ)
num_y_pos = math.ceil(H/SZ)

positions = []

for y in range(num_y_pos):
    for x in range(num_x_pos):
        y_pos, x_pos = SZ*y,SZ*x
        positions.append((x_pos,y_pos))

panorama = IP.WindowableRGBImage(W,H)

for xpos, ypos in positions:
    window = get_window(xpos,ypos,SZ,SZ)
    if np.all(window==0):
        print("Windows is entirely 0")
        continue
    LAB = cv2.cvtColor(window,cv2.COLOR_RGB2LAB)
    L, A, B = (IP.rescale_array(LAB[:,:,c] for c in range(3)))
    print("Quantizing...")
    centers,labels2D,qimg = IP.unique_quantize(gs,2,ignore_zero=True)
    print("Done.")
    gs_mask = labels2D == 1
    gs = np.where(gs_mask,1.0,0.0).astype(float)
    batch_X = np.array([np.expand_dims(gs,axis=2)],float)
    batch_predictedY = model.predict(batch_X)
    out_plot = IP.rescale_array(np.squeeze(batch_predictedY[0]))
    naive_out_mask = IP.binarize_otsu(out_plot)

    # out_mask = out_plot > np.mean(out_plot[~naive_out_mask])

    out_mask = naive_out_mask

    out_mask_rgb = IP.greyscale_plot_to_color_image(np.where(out_mask,1.0,0.0).astype(float))

    print(out_mask_rgb.dtype,out_mask_rgb.shape)

    seed_regions = IP.BooleanImageOps.connected_components(out_mask,4)

    full_regions = IP.BooleanImageOps.connected_components(gs_mask, 4)

    seed_regions_image = IP.grouping_to_color_image(seed_regions)

    seed_region_pixel_locations = np.transpose(np.nonzero(seed_regions!=-1))

    seed_regions_tree = KDTree(seed_region_pixel_locations)

    voronoi_grouping = np.zeros_like(seed_regions) - 1

    number_of_ungrouped_pixels = 0

    for r, c in np.transpose(np.nonzero(gs_mask)):
        if seed_regions[r,c] == -1:
            d, i = seed_regions_tree.query((r,c),1)
            nearest_pixel_location = seed_region_pixel_locations[i]
            # only label the pixel if the target pixel and seed pixel lie within the same group on the onprocessed mask
            if full_regions[r,c] == full_regions[nearest_pixel_location[0],nearest_pixel_location[1]]:
                voronoi_grouping[r,c] = seed_regions[nearest_pixel_location[0],nearest_pixel_location[1]]
            else:
                number_of_ungrouped_pixels += 1
        else:
            voronoi_grouping[r,c] = seed_regions[r,c]

    print(f"# Ungrouped pixels: {number_of_ungrouped_pixels}")

    voronoi_grouping_image = IP.grouping_to_color_image(voronoi_grouping)

    conv_result = scipy.ndimage.convolve(voronoi_grouping,np.ones((3,3)),mode='constant',cval = -1)

    edge_map = np.zeros_like(conv_result).astype(bool)

    for x in range(edge_map.shape[1]):
        for y in range(edge_map.shape[0]):
            if conv_result[y,x] != 9*voronoi_grouping[y,x]:
                edge_map[y,x] = True

    voronoi_grouping_image = window.copy()

    voronoi_grouping_image[edge_map,...] = 0
            
    panorama.write_image(int(xpos),int(ypos),voronoi_grouping_image)

    PIL.Image.fromarray(IP.ImageStrip(

        IP.greyscale_plot_to_color_image(IP.rescale_array(gs)),
        IP.greyscale_plot_to_color_image(IP.rescale_array(out_plot)),
        IP.greyscale_plot_to_color_image(out_mask),
        
        voronoi_grouping_image,max_cols=2
    ).getImagePixels()).save(f"debug/panorama_snippet_results/x={xpos}_y={ypos}.png")

IP.imshow(panorama.getImagePixels(),"Panorama Output Mask")