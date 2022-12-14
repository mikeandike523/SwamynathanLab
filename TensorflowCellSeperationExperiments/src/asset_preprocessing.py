import PIL.Image
import numpy as np
import cv2
import alphashape
import shapely.geometry
import rasterio.features

PIL.Image.MAX_IMAGE_PIXELS = None

REQUIRED_RESAMPLE = 0.25

import image_processing as IP
import utils as U

def main():

    U.init_folder('debug')

    U.dprint("Starting asset preprocessing...",False)

    # import argparse
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--filepath")

    # args = parser.parse_args()

    # filepath = args.filepath

    filepath = "../assets/golden_retina_map.png"

    pixels = np.array(PIL.Image.open(filepath))

    pixels_unscaled = pixels.copy()

    init_H, init_W = pixels.shape[:2]

    IP.imshow(pixels,"original_image_unscaled")

    if REQUIRED_RESAMPLE != 1.0:

        pixels=IP.scale_image_nearest(pixels,REQUIRED_RESAMPLE)

    H, W = pixels.shape[:2]

    IP.imshow(pixels,"original_image")

    pixels_gs = np.mean(pixels.astype(float)/255,axis=2)

    LAB_image = cv2.cvtColor(pixels,cv2.COLOR_RGB2LAB)

    L, A, B = [LAB_image[:,:,c] for c in range(3)]
    L = IP.rescale_array(L)
    A = IP.rescale_array(A)
    B = IP.rescale_array(B)
    
    figure = IP.ImageStrip(
        IP.greyscale_plot_to_color_image(L),
        IP.greyscale_plot_to_color_image(A),
        IP.greyscale_plot_to_color_image(B),
        max_cols=None)

    
    IP.imshow(figure.getImagePixels(),"LAB channels")

    U.dprint("Quantizing...",True)

    centers,labels2D,qimg = IP.unique_quantize(B,8)

    IP.imshow(IP.greyscale_plot_to_color_image(qimg),"qimg")

    U.dprint("Creating grouping image...",True)

    IP.imshow(IP.grouping_to_color_image(labels2D),"labels2D")

    mask = labels2D == 1

    IP.imshow(IP.greyscale_plot_to_color_image(mask),"Mask")

    """Clean the mask"""

    # grouping = IP.dbscan_group_cityblock(mask,3)
    
    # IP.imshow(IP.grouping_to_color_image(grouping),"grouping")
    
    # clean_grouping = IP.Grouping.filter_all_but_k_largest_islands(grouping,1)

    U.dprint("Finding islands...", True)

    # grouping = IP.BooleanImageOps.connected_components(mask,8)

    grouping = IP.dbscan_group_cityblock(mask, 7)

    U.dprint("Filtering islands", True)

    clean_grouping = IP.Grouping.filter_small_islands_by_pixel_count(grouping,500)
    
    U.dprint("Creating clean grouping image...", True)

    IP.imshow(IP.grouping_to_color_image(clean_grouping),"clean grouping")

    mask = clean_grouping != -1

    mask = IP.BooleanImageOps.closeWithKernel(mask,2,2,2)

    IP.imshow(IP.greyscale_plot_to_color_image(mask),"Clean Mask")

    """perform a second-pass of cleanup"""

    U.dprint("Performing second-pass cleanup.",True)

    U.dprint("Running opencv connected_components...",True)

    grouping = IP.BooleanImageOps.connected_components(mask,8)

    # grouping = IP.dbscan_group_cityblock(mask,1)

    U.dprint("Filtering small islands...")

    clean_grouping = IP.Grouping.filter_small_islands_by_pixel_count(grouping,5)

    mask = clean_grouping != -1

    mask = IP.BooleanImageOps.dilateWithKernel(mask,2,2)

    U.dprint("Wrapping with alphashape.",True)

    nonzero_locations = np.transpose(np.nonzero(mask))

    nonzero_locations = np.flip(nonzero_locations, axis=1) # convert r,c to x,y

    ashape = alphashape.alphashape(nonzero_locations,0.25)

    filled_mask = np.zeros_like(mask)

    def add_polygon_to_filled_mask(polygon_obj):
        nonlocal filled_mask
        msk = rasterio.features.rasterize([polygon_obj],out_shape=filled_mask.shape)
        filled_mask = np.logical_or(filled_mask,msk)

    if isinstance(ashape,shapely.geometry.multipolygon.MultiPolygon):

        for geom in ashape.geoms:
            add_polygon_to_filled_mask(geom)

    if isinstance(ashape,shapely.geometry.polygon.Polygon):
        add_polygon_to_filled_mask(ashape)

    border = IP.BooleanImageOps.get_longest_contour(filled_mask)

    border = IP.Curve.resample_by_segment_length(border,25,True)

    border = IP.Curve.fit_univariate_spline(border)

    filled_mask = IP.Curve.curve_to_mask(border,W,H)

    large_filled_mask_image = cv2.resize(IP.greyscale_plot_to_color_image(filled_mask),dsize=(init_W,init_H),interpolation=cv2.INTER_NEAREST)

    large_filled_mask = large_filled_mask_image[:,:,0] > 0

    unscaled_masked_image = IP.BooleanImageOps.apply_mask_to_image(large_filled_mask,pixels_unscaled)

    IP.imshow(unscaled_masked_image,"Masked Image")

    masked_image_visualization = pixels_unscaled.copy()

    image_greyscale = np.dstack((np.mean(pixels_unscaled.astype(float),axis=2),)*3).astype(np.uint8)

    masked_image_visualization[np.logical_not(large_filled_mask),...] = image_greyscale[np.logical_not(large_filled_mask),...]

    IP.imshow(masked_image_visualization, "Mask Visualization")

    IP.imshow(IP.greyscale_plot_to_color_image(large_filled_mask),"Filled Mask")
        

if __name__ == '__main__':
    main()
