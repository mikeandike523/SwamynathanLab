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
from core.performance import ReportDuration
from core.progress import Progress
from core.analysis import spine as spine_analysis
from core.io import imsave
from core.path import init_folder

import utils as U
import image_processing as IP
from simplefilepicker import askopenfilename


""" --- Adapted from D:\SwamynathanLab\TensorflowCellSeperationExperiments\src\cell_count_retinal_mapping\mapping_with_CNN.py"""

from core.deeplearning import CellDifferentiator

pretrained_path = "D:\\SwamynathanLab\\Retina Mapping Project\\output\\segmentation.h5"

differentiator = CellDifferentiator(128,128,pretrained_path)

def run_differentiator(pixels):
    retval =differentiator.apply_to_images(np.array([pixels]))[0]
    return retval

def get_neural_network_output(masked_image_white_background):

    H,W = masked_image_white_background.shape[:2]

    windowable_input_image = IP.to_WindowableRGBImage(masked_image_white_background)

    windowable_output_image = IP.to_WindowableRGBImage(np.zeros_like(masked_image_white_background))

    locations = U.Geometry.get_window_locations_covering_image(
        W,H,128,128
    )

    for x,y in locations:
        input_window = windowable_input_image.read_window(x,y,128,128)
        output_window = run_differentiator(input_window)
        windowable_output_image.write_window(x,y,128,128,np.array(output_window,np.uint8))

    return windowable_output_image.getImagePixels()

def threshold_network_output(output_pixels):
    return np.mean(output_pixels.astype(float)/255,axis=2) > 0.5

def predict_labels(masked_image_white_background, conservative_cell_mask):

    output_pixels = get_neural_network_output(masked_image_white_background)

    mask = threshold_network_output(output_pixels)

    mask=np.logical_and(mask,conservative_cell_mask)

    labels = IP.BooleanImageOps.connected_components(mask,4)

    return labels

def predict_labels_from_output(neural_network_output, conservative_cell_mask):

    mask = threshold_network_output(neural_network_output)

    mask=np.logical_and(mask,conservative_cell_mask)

    labels = IP.BooleanImageOps.connected_components(mask,4)

    return labels

""" --- """

""" --- Adapted from D:\SwamynathanLab\ImageJCloneWithPython\plugins\mark_cells\masking.py """

def get_conservative_cell_mask(masked_image_white_background):

    image_LAB = cv2.cvtColor(masked_image_white_background,cv2.COLOR_RGB2LAB)

    L= U.rescale_array(image_LAB[:,:,0].astype(float))
    A= U.rescale_array(image_LAB[:,:,1].astype(float))
    B= U.rescale_array(image_LAB[:,:,2].astype(float))

    channel = 1.0-L

    centers, labels2D, qimg = IP.unique_quantize(channel,2,True)

    conservative_cell_mask = labels2D == 1

    adaptive_cell_mask = IP.Channel.adaptive_threshold_by_mean(channel,21)

    conservative_cell_mask = np.logical_and(conservative_cell_mask,adaptive_cell_mask)

    element = IP.circular_structuring_element(1,int,remove_center=True)
    
    ncounts = scipy.ndimage.convolve(conservative_cell_mask.astype(int),element)

    conservative_cell_mask[ncounts<2] = False

    return conservative_cell_mask

""" --- """

""" --- Adapted from D:\SwamynathanLab\ImageJCloneWithPython\main.py"""
     
def perform_watershed(input_image,seed_grouping, conservative_cell_mask):

    LAB_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2LAB)

    L_channel = U.rescale_array(LAB_image[:,:,0])
    # A_channel = U.rescale_array(LAB_image[:,:,1])
    # B_channel = U.rescale_array(LAB_image[:,:,2])

    watershed_channel = IP.greyscale_plot_to_color_image(L_channel)

    seed_grouping_watershed = IP.Grouping.watershed(conservative_cell_mask, seed_grouping,watershed_channel)

    return seed_grouping_watershed

""" --- """


# Application Parameters
GROUP_PIXEL_LENGTH = 200
GROUP_PIXEL_RESOLUTION = 10
# CLEAN_MASK = fluiddict(
#     FG_MIN_PIXELS = 20*20,
#     BG_MIN_PIXELS = 40*40,
#     DS=2
# )
CELL_MASK_DILATE_R = 4
CELL_MASK_DILATE_ITERS = 2
CLEAN_MASK_BY_PROXIMITY = fluiddict(
    DS = 1,
    PATCH_SIZE = 256,
    PROXIMITY=7,
    ISLAND_MIN_PIXELS = 750
)
# CLOSE_MASK = fluiddict(
#     R=4, # circular structuring element radius
#     ITERS=2, # 1 = dilate-erode, 2- dilate-dilate-erode-erode, etc.
#     REPEATS=2 # number of times to perform closure operation
# )
WRAP_ISLANDS = fluiddict(
    ALPHA=0.125,
    DS = 2
)
FIND_SPINE = fluiddict(
    DS = 4
)
CUT_MASK_INTO_GROUPS = fluiddict(
    DS = 4
)
ANNOTATION_OFFSET = 400

# Helper functions

def create_image_and_mask_comparison(image,mask):
    return IP.Image.mix([IP.greyscale_plot_to_color_image(mask),image],[0.75,0.25])

cm = CheckpointManager(name="main.py")

def program_setup(selection):
    
    if selection in ["f","r"]:
        init_folder("debug",clear=True)

@cm.checkpoint
def cpt_choose_file(db: Database):

    panorama_path = askopenfilename()

    if not panorama_path:

        print("User exited choose-file dialog.")
        return cm.stop_here()

    db.set_variable("panorama_path",panorama_path, direct=True) 

@cm.checkpoint
def cpt_load_panorama(db: Database):

    panorama_image = np.asarray(PIL.Image.open(db.get_variable("panorama_path")))
    db.set_variable("panorama_image",panorama_image)
    db.set_variable("panorama_shape",panorama_image.shape)
    imshow(panorama_image,"Panorama Image")

@cm.checkpoint
def cpt_convert_to_LAB(db: Database):

    db.set_variable("panorama_image_LAB",cv2.cvtColor(db.get_variable("panorama_image"),cv2.COLOR_RGB2LAB))
    
@cm.checkpoint    
def cpt_obtain_B_channel(db: Database):
    
    panorama_image_LAB = db.get_variable("panorama_image_LAB")
    B_channel = U.rescale_array(panorama_image_LAB[:,:,2].astype(float))
    db.set_variable("B_channel",B_channel)
    imshow(IP.greyscale_plot_to_color_image(B_channel),"B_channel")

@cm.checkpoint
def cpt_filter_B_channel(db: Database):

    B_channel = db.get_variable("B_channel")
    centers, labels2D, qimg = IP.unique_quantize(B_channel, 7, ignore_zero = False)
    cell_mask = labels2D < 1
    
    cell_mask = IP.BooleanImageOps.dilateWithKernel(cell_mask,CELL_MASK_DILATE_R,CELL_MASK_DILATE_ITERS)
    
    imshow(IP.grouping_to_color_image(labels2D),"Quantized B Channel Levels")
    B_channel[~cell_mask]=0.0
    imshow(IP.greyscale_plot_to_color_image(B_channel),"Filtered B Channel")
    db.set_variable("filtered_B_channel",B_channel)
    db.set_variable("cell_mask", cell_mask)

@cm.checkpoint
def cpt_obtain_L_channel(db: Database):

    panorama_image_LAB = db.get_variable("panorama_image_LAB")
    L_channel = U.rescale_array(panorama_image_LAB[:,:,0].astype(float))
    db.set_variable("L_channel", L_channel)

@cm.checkpoint
def cpt_mask_L_channel(db: Database):

    L_channel = db.get_variable("L_channel")
    cell_mask = db.get_variable("cell_mask")
    masked_L_channel = np.where(cell_mask, L_channel,0.0).astype(float)
    db.set_variable("masked_L_channel", masked_L_channel)
    imshow(IP.greyscale_plot_to_color_image(masked_L_channel),"Masked L Channel")

@cm.checkpoint
def cpt_quantize_masked_L_channel(db: Database):
    
    masked_L_channel = db.get_variable("masked_L_channel")

    centers, labels2D, qimg = IP.unique_quantize(masked_L_channel,2,ignore_zero=2)

    imshow(IP.grouping_to_color_image(labels2D),"Quantized Masked L Channel Labels")
    
    cell_mask_2 = labels2D == 0

    imshow(IP.greyscale_plot_to_color_image(cell_mask_2),"Cell Mask 2")

    db.set_variable("cell_mask_2", cell_mask_2)
    
    db.set_variable("cell_mask_3", cell_mask_2)

# @cm.checkpoint
# def cpt_clean_cell_mask_2(db: Database):
#     ds = CLEAN_MASK.DS
#     cell_mask_2 = db.get_variable("cell_mask_2")
#     # H, W = cell_mask_2.shape
#     # cell_mask_2 = IP.BooleanImageOps.scale_by(cell_mask_2, 1/ds)
#     # with ReportDuration("Cleaning by pixel count"):
#     #     cell_mask_3 = IP.BooleanImageOps.clean_by_pixel_count(cell_mask_2,CLEAN_MASK.FG_MIN_PIXELS/(ds**2), CLEAN_MASK.BG_MIN_PIXELS/(ds**2),connectivity=8)
#     # cell_mask_3 = IP.BooleanImageOps.resize_to(cell_mask_3,W,H)
#     db.set_variable("cell_mask_3",cell_mask_3, direct=False)
#     cell_mask_3 = cell_mask_2
#     imshow(IP.greyscale_plot_to_color_image(cell_mask_3),"Cell Mask 3")

@cm.checkpoint
def cpt_clean_cell_mask_by_proximity(db: Database):

    H, W = db.get_variable("panorama_shape")[:2]

    cell_mask_3 = db.get_variable("cell_mask_3")
    
    cell_mask_3 = IP.BooleanImageOps.scale_by(cell_mask_3,1/CLEAN_MASK_BY_PROXIMITY.DS)

    with ReportDuration(f"Running dbscan with eps={CLEAN_MASK_BY_PROXIMITY.PROXIMITY}"):
        grouping = IP.dbscan_group_cityblock_in_patches(cell_mask_3,CLEAN_MASK_BY_PROXIMITY.PATCH_SIZE
                                                        ,CLEAN_MASK_BY_PROXIMITY.PROXIMITY/CLEAN_MASK_BY_PROXIMITY.DS)
    
    with ReportDuration("Filtering small islands"):
        grouping = IP.Grouping.filter_small_islands_by_pixel_count(grouping, CLEAN_MASK_BY_PROXIMITY.ISLAND_MIN_PIXELS)

    cell_mask_4 = grouping != -1

    cell_mask_4 = IP.BooleanImageOps.resize_to(cell_mask_4,W,H)

    db.set_variable("cell_mask_4",cell_mask_4,direct=False)
    
    db.set_variable("cell_mask_5",cell_mask_4,direct=False)

    imshow(IP.greyscale_plot_to_color_image(cell_mask_4),"Cell Mask 4")

# @cm.checkpoint
# def morph_clean_mask(db: Database):
#     cell_mask_4 = db.get_variable("cell_mask_4")
#     pr = Progress("close_mask")
#     # For now, just do closure, in the future, can try doing opening as well
#     with pr.track("close_mask"):
#         cell_mask_5 = IP.BooleanImageOps.closeWithKernel(cell_mask_4,CLOSE_MASK.R,CLOSE_MASK.ITERS,CLOSE_MASK.REPEATS)
#     db.set_variable("cell_mask_5",cell_mask_5)
#     imshow(IP.greyscale_plot_to_color_image(cell_mask_5),"Final Cell Mask")
# @cm.checkpoint
# def cpt_clean_cell_mask_3(db: Database):
#     ds = CLEAN_MASK.DS
#     cell_mask_5 = db.get_variable("cell_mask_5")
#     H, W = db.get_variable("panorama_shape")[:2]
#     cell_mask_5 = IP.BooleanImageOps.scale_by(cell_mask_5, 1/ds)
#     with ReportDuration("Cleaning by pixel count"):
#         cell_mask_6 = IP.BooleanImageOps.clean_by_pixel_count(cell_mask_5,CLEAN_MASK.FG_MIN_PIXELS/(ds**2), CLEAN_MASK.BG_MIN_PIXELS/(ds**2),connectivity=8)
#     cell_mask_6 = IP.BooleanImageOps.resize_to(cell_mask_6,W,H)
#     db.set_variable("cell_mask_5",cell_mask_6)
#     imshow(IP.greyscale_plot_to_color_image(cell_mask_5),"Final Cell Mask")

@cm.checkpoint
def get_alpha_shapes(db: Database):

    pr=Progress("get_alpha_shapes")

    with pr.track("read database"):
        cell_mask_5 = db.get_variable("cell_mask_5")

    H, W = cell_mask_5.shape

    ds = WRAP_ISLANDS.DS

    cell_mask_5 = IP.BooleanImageOps.scale_by(cell_mask_5, 1/ds)

    with pr.track("get nonzero locations"):
        pixel_locations_xy = np.flip(np.transpose(np.nonzero(cell_mask_5)),axis=1)

    with pr.track("get alphashapes"):
        polys = IP.run_alphashape_and_get_polygons(pixel_locations_xy,WRAP_ISLANDS.ALPHA)

    with pr.track("write database"):

        db.set_variable("island_polys",polys) #@TODO: double check that variables are indirect by default

    cell_layer_mask = np.zeros_like(cell_mask_5)

    HH, WW = cell_layer_mask.shape

    for poly in polys:

        curve = np.squeeze(poly.boundary.coords).astype(int)

        with pr.track("creating mask for alphashape polyon"):

            mask = IP.Curve.curve_to_mask(curve,WW,HH)

        cell_layer_mask = np.logical_or(cell_layer_mask,mask)

    cell_layer_mask = IP.BooleanImageOps.resize_to(cell_layer_mask,W,H)

    db.set_variable("cell_layer_mask",cell_layer_mask)

    imshow(IP.greyscale_plot_to_color_image(cell_layer_mask),"Cell Layer Mask")

    # create cell layer mask visualization
    cell_layer_mask_image = IP.greyscale_plot_to_color_image(cell_layer_mask)
    panorama_image = db.get_variable("panorama_image")
    mixed_image = IP.Image.mix([cell_layer_mask_image,panorama_image],[0.75,0.25])

    imshow(mixed_image,"Cell Layer Mask Visualization")

@cm.checkpoint
def isolate_cell_layer_wings(db: Database):

    cell_layer_mask = db.get_variable("cell_layer_mask")
    
    pr = Progress("isolate_cell_layer_wings")

    with pr.track("connected components"):
        grouping = IP.BooleanImageOps.connected_components(cell_layer_mask,connectivity=8)

    components = IP.Grouping.components_from_grouping(grouping)

    if len(components) != 2:
        raise ValueError("More than two retina wings detected. ")

    db.set_variable("cell_layer_wings",components)

def get_boundaries_and_spines(db: Database, proxy: MapProxy):

    pr = Progress("get_boundaries_and_spines")

    wing_mask = proxy.get_variable("cell_layer_wings")

    H, W = wing_mask.shape

    wing_mask = IP.BooleanImageOps.scale_by(wing_mask,1/FIND_SPINE.DS)

    boundary = spine_analysis.get_boundary(wing_mask)

    naive_centerline = spine_analysis.get_naive_centerline(boundary)

    true_spine = spine_analysis.convert_naive_spine_to_real_spine(boundary,naive_centerline)

    boundary = (FIND_SPINE.DS*boundary.astype(float)).astype(int)

    naive_centerline = (FIND_SPINE.DS*naive_centerline.astype(float)).astype(int)

    true_spine = (FIND_SPINE.DS*true_spine.astype(float)).astype(int)

    proxy.set_variable("wing_boundaries",boundary)

    proxy.set_variable("wing_naive_centerlines",naive_centerline)

    proxy.set_variable("true_spines",true_spine)

    curve_image = np.full((H,W,3),255,np.uint8)

    pallete = IP.Pallete()
    
    with pr.track("Visualizing boundaries and spines..."):

        curve_image = IP.Image.draw_curve_on(curve_image,boundary,pallete.nextColor(),2,closed=True)

        curve_image = IP.Image.draw_curve_on(curve_image,naive_centerline,pallete.nextColor(),2,closed=False)

        curve_image = IP.Image.draw_curve_on(curve_image,true_spine,pallete.nextColor(),2,closed=False)

    imshow(curve_image,f"Boundary and Centerline -- Wing {proxy.index()+1} of {proxy.size()}")

cm.register_mapped_checkpoint("cell_layer_wings",get_boundaries_and_spines)

def cut_mask_into_groups(db: Database, proxy: MapProxy):\
    
    pr = Progress("cut_mask_into_groups")
    
    panorama_image = db.get_variable("panorama_image")

    H, W = db.get_variable("panorama_shape")[:2]

    wing_mask = proxy.get_variable("cell_layer_wings")

    wing_mask = IP.BooleanImageOps.scale_by(wing_mask,1/CUT_MASK_INTO_GROUPS.DS)

    nonzero_locations_xy = IP.BooleanImageOps.where2D(wing_mask)

    boundary = proxy.get_variable("wing_boundaries")

    boundary = (boundary.astype(float)/CUT_MASK_INTO_GROUPS.DS).astype(int)

    true_spine = proxy.get_variable("true_spines")

    true_spine = (true_spine.astype(float)/CUT_MASK_INTO_GROUPS.DS).astype(int)

    group_curve = IP.Curve.resample_by_segment_length(true_spine,GROUP_PIXEL_RESOLUTION,False)

    segments_per_group = GROUP_PIXEL_LENGTH/GROUP_PIXEL_RESOLUTION

    normals,tangents = IP.Curve.curve_tangents_and_normals(group_curve)

    group_masks = []
    
    num_groups = int(len(group_curve)//segments_per_group)

    segment_offs = (len(group_curve)%segments_per_group)//2
    
    # segment_offs = 0 
    
    print(len(group_curve)-segments_per_group*num_groups)

    spine_tree = KDTree(group_curve)
    
    for group_number in range(num_groups):
        
        print(f"Populating mask for group {group_number+1} of {num_groups}...")

        group_mask = np.zeros_like(wing_mask)
        
        with pr.track(f"Creating manual annotation data for wing {proxy.index()} group {group_number}..."):
            image_for_manual_annotation = IP.BooleanImageOps.select_image_region_by_mask(panorama_image,group_mask,(255,255,255))
            imsave(image_for_manual_annotation,f"output/images_for_manual_annotation/{group_number}.png")
            
        iA = segment_offs+group_number*segments_per_group
        iB = segment_offs+(group_number+1)*segments_per_group

        print(group_number, num_groups, iA,iB,len(group_curve), min(iB,len(group_curve)-1)-min(iA,len(group_curve)-1))

        for nonzero_location in nonzero_locations_xy:
            d, i = spine_tree.query(nonzero_location,1)
            x,y = nonzero_location
            if i >= iA and i <= iB:
                group_mask[y,x] = True

        group_mask = IP.BooleanImageOps.resize_to(group_mask,W,H)

        group_masks.append(group_mask)

    proxy.set_variable("group_masks",group_masks)

    group_mask_visualization = np.zeros((H,W),int)-1

    for i, mask in enumerate(group_masks):
        group_mask_visualization[mask] = i

    proxy.set_variable("group_curves",(CUT_MASK_INTO_GROUPS.DS * group_curve).astype(int))
    
    proxy.set_variable("num_groups", num_groups)
    
    proxy.set_variable("segments_per_group",segments_per_group)
    
    proxy.set_variable("segment_offs",segment_offs)

    imshow(IP.Grouping.to_color_with_default_pallete(group_mask_visualization),f"Group Masks -- Wing {proxy.index()+1}")

cm.register_mapped_checkpoint("cell_layer_wings", cut_mask_into_groups)

def evaluate_counts_over_groups(db: Database, proxy: MapProxy):
    pr = Progress("evaluate_counts_over_groups")

    H, W = db.get_variable("panorama_shape")[:2]

    panorama_image = db.get_variable("panorama_image")

    group_masks=proxy.get_variable("group_masks")

    vis_conservative_cell_mask = np.zeros((H,W),bool)
    
    vis_watershed_labels = np.zeros((H,W),int) - 1
    
    vis_seed_labels = np.zeros((H,W),int) - 1

    vis_neural_network_output = np.zeros((H,W,3),np.uint8)

    group_counts = []

    for i, group_mask in enumerate(group_masks):

        print(f"Processing group {i+1} of {len(group_masks)}")

        print(np.any(group_mask))
        
        if not np.any(group_mask):
            raise ValueError("Group mask is empty.")

        extended_group_mask = np.copy(group_mask)
        
        if i > 0:
            
            previous_group_mask = group_masks[i-1]
            extended_group_mask = np.logical_or(extended_group_mask,previous_group_mask)
            
            if i < len(group_masks) - 1:
                next_group_mask = group_masks[i+1]
                extended_group_mask = np.logical_or(extended_group_mask,next_group_mask)
            
            
            top_left_corner,bottom_right_corner = IP.BooleanImageOps.bounds(group_mask)
            group_slice = np.s_[top_left_corner[1]:(bottom_right_corner[1]+1),top_left_corner[0]:(bottom_right_corner[0]+1)]
            new_extended_group_mask = np.zeros_like(extended_group_mask)
            new_extended_group_mask[group_slice] = extended_group_mask[group_slice]
            extended_group_mask = new_extended_group_mask

        top_left_corner,bottom_right_corner = IP.BooleanImageOps.bounds(extended_group_mask)
        
        clipped_mask = IP.BooleanImageOps.clipped_mask(extended_group_mask)

        group_slice = np.s_[top_left_corner[1]:(bottom_right_corner[1]+1),top_left_corner[0]:(bottom_right_corner[0]+1)]

        WW = bottom_right_corner[0]-top_left_corner[0]+1
        HH = bottom_right_corner[1]-top_left_corner[1]+1

        image_section = panorama_image[group_slice]

        masked_image_section=IP.BooleanImageOps.apply_mask_to_image(clipped_mask,image_section, (255,255,255))

        conservative_cell_mask = get_conservative_cell_mask(masked_image_section)

        vis_conservative_cell_mask[group_slice] = conservative_cell_mask

        neural_network_output = get_neural_network_output(masked_image_section)

        vis_neural_network_output[group_slice] = neural_network_output

        seed_grouping = predict_labels_from_output(neural_network_output,conservative_cell_mask)
        
        vis_seed_labels[group_slice] = seed_grouping
        
        watershed_labels = perform_watershed(masked_image_section,seed_grouping,conservative_cell_mask)
        
        vis_watershed_labels[group_slice] = watershed_labels
        
        group_counts.append(np.max(vis_watershed_labels)+1)

        proxy.set_variable("group_counts",group_counts)
        
    proxy.set_variable("vis_watershed_labels",vis_watershed_labels)

    with pr.track("Generating visualizations..."):

        imshow(IP.greyscale_plot_to_color_image(vis_conservative_cell_mask),f"Conservative Cell Mask wing {proxy.index()+1}.")

        imshow(vis_neural_network_output,f"Nueral network output wing {proxy.index()+1}")
        
        imshow(IP.grouping_to_color_image(vis_watershed_labels), f"Watershed Labels Wing {proxy.index()+1}")
        
        imshow(IP.grouping_to_color_image(vis_seed_labels), f"Seed labels Wing {proxy.index()+1}")

cm.register_mapped_checkpoint("cell_layer_wings", evaluate_counts_over_groups)

@cm.checkpoint
def assemble_results(db:Database):
    
    H, W = db.get_variable("panorama_shape")[:2]
    
    cell_layer_mask = db.get_variable("cell_layer_mask")
    
    wing_masks = db.get_variable("cell_layer_wings")
    
    num_wings = len(wing_masks)
    
    panorama_image = db.get_variable("panorama_image")
    
    plot = IP.BooleanImageOps.apply_mask_to_image(cell_layer_mask, panorama_image, (0,0,0))
    
    true_spines = db.get_variable("true_spines")
    
    group_counts = db.get_variable("group_counts")
    
    group_masks = db.get_variable("group_masks")
    
    edge_plot_grouping = np.zeros_like(cell_layer_mask).astype(int) -1
    
    vis_watershed_labels = db.get_variable("vis_watershed_labels")
    
    print(vis_watershed_labels)
    
    all_watershed_labels = np.zeros((H,W),int)-1
    
    for wing in range(num_wings):
        print(f"Assembling results for wing {wing+1}")
        
        group_curve = db.get_variable("group_curves")[wing]
        
        segments_per_group = GROUP_PIXEL_LENGTH/GROUP_PIXEL_RESOLUTION

        tagents,normals = IP.Curve.curve_tangents_and_normals(group_curve)
    
        num_groups = db.get_variable("num_groups")[wing]
        
        segment_offs = db.get_variable("segment_offs")[wing]
        
        wing_group_counts = group_counts[wing]
        
        edge_plot_group_number = 0
        
        wing_group_masks = group_masks[wing]
        
        print(f"num groups: {num_groups}")
        
        wing_watershed_labels = vis_watershed_labels[wing]
        
        for group_number in range(num_groups):
            
            group_mask = wing_group_masks[group_number]
            
            edge_plot_grouping[group_mask] = edge_plot_group_number
            
            iM = int(segment_offs + (group_number+0.5)*segments_per_group)
            
            M = group_curve[iM]
            
            N = U.Geometry.normalizeVector(normals[iM])
            
            annotation_position = (M+N*ANNOTATION_OFFSET).astype(int)
            
            group_count = wing_group_counts[group_number]
            
            plot = IP.Draw.draw_annotation(plot,f"{int(group_count)}",annotation_position[0],annotation_position[1],5,(255,0,0),4,0.75,4)
    
            edge_plot_group_number += 1
    
        mask = wing_watershed_labels!=-1
        all_watershed_labels[mask] = wing_watershed_labels[mask]
    
    imshow(IP.grouping_to_color_image(all_watershed_labels),"All watershed labels")
    
    edge_mask = IP.Grouping.get_edge_mask_from_grouping(edge_plot_grouping,3)
    
    imshow(IP.Grouping.to_color_with_default_pallete(edge_plot_grouping),"edge plot grouping")
    
    plot = IP.BooleanImageOps.apply_mask_to_image(~edge_mask,plot,(0,255,0))     
         
         
         
    watershed_label_edge_mask = IP.Grouping.get_edge_mask_from_grouping(all_watershed_labels,1)

    plot = IP.BooleanImageOps.apply_mask_to_image(~watershed_label_edge_mask,plot,(0,255,0))     
            
    imshow(plot,"Group Counts Plot")

cm.menu(program_setup)