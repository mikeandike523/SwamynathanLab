# Todo list:
# Plan to switch numba or numpy based windowable images to io3 images
# Most relevant function to change is `get_neural_network_output`

from core import setup
setup.setup()

import PIL.Image
import numpy as np
import cv2
from scipy.spatial import KDTree
import scipy.ndimage
from core.checkpoint import CheckpointManager
from core.database import Database, MapProxy
from core.debugging import imshow, Logging
from core.types import fluiddict
from core.progress import Progress
from core.io import imload, imsave
from core.path import init_folder, remove_fs_significant_chars
from core.debugging import SequentialNames
from core.analysis import spine as spine_analysis
import utils as U
import image_processing as IP
from core.file_picker import askopenfilename
import functools
import os
import click
from io3.pycore.image import Image as Io3Image
from core.imdebug_server import ImDebugServer

if __name__ == "__main__":
    im_debug_server = ImDebugServer()

    im_debug_server.start()

    """ --- Adapted from D:\SwamynathanLab\TensorflowCellSeperationExperiments\src\cell_count_retinal_mapping\mapping_with_CNN.py"""

    from core.deeplearning import CellDifferentiator

    pretrained_path = os.path.realpath("assets\\segmentation.h5")

    print(f"Pretrained path: {pretrained_path}")

    SZ = 64

    differentiator = CellDifferentiator(SZ,256,pretrained_path)

    def run_differentiator(pixels):
        retval =differentiator.apply_to_images(np.array([pixels]))[0]
        return retval

    def get_neural_network_output(masked_image_white_background):

        H,W = masked_image_white_background.shape[:2]

        # windowable_input_image = IP.to_WindowableRGBImage(masked_image_white_background)
        windowable_input_image = Io3Image.from_array(masked_image_white_background)

        im_debug_server.imshow(windowable_input_image.to_array())

        # windowable_output_image = IP.to_WindowableRGBImage(np.zeros_like(masked_image_white_background))
        windowable_output_image = Io3Image.from_array(np.zeros_like(masked_image_white_background))

        locations = U.Geometry.get_window_locations_covering_image(
            W,H,SZ,SZ
        )

        for x,y in locations:
            # input_window = windowable_input_image.read_window(x,y,SZ,SZ)
            input_window = windowable_input_image.get_window(x,y,SZ,SZ)

            output_window = run_differentiator(input_window.to_array())
            # windowable_output_image.write_window(x,y,SZ,SZ,np.array(output_window,np.uint8))
            windowable_output_image.set_window(x,y,Io3Image.from_array(np.array(output_window,np.uint8)))

        # return windowable_output_image.getImagePixels()

        return windowable_output_image.to_array() # this function will call the swap_back automatically

    def threshold_network_output(output_pixels):
        # return np.mean(output_pixels.astype(float)/255,axis=2) > 0.5
        return IP.binarize_otsu(np.mean(output_pixels.astype(float),axis=2)/255).astype(float) > 0.5

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

    GROUP_PIXEL_LENGTH = 200

    GROUP_PIXEL_RESOLUTION = 10

    WRAP_ISLANDS = fluiddict(
        ALPHA=0.075,
        DS = 2
    )

    FIND_SPINE = fluiddict(
        DS = 4
    )

    CUT_MASK_INTO_GROUPS = fluiddict(
        DS = 4
    )

    ANNOTATION_OFFSET = 400

    seq_cell_mask = SequentialNames("cell_mask")

    cm = CheckpointManager("pipeline.py")

    if click.confirm("Clear debug and output folders?", default=False):

        init_folder("debug",clear=True)
        init_folder("output",clear=True)
            
    def pick_file(db: Database): # Not a checkpoint. Should run once each time the program runs
        
        panorama_path = askopenfilename()
        
        if panorama_path is None or not panorama_path:
            print("No panorama selected. Quitting...") 
            exit()
            
        subdb_name = remove_fs_significant_chars(panorama_path).replace(" ","_")

        db.engage_subdb(subdb_name)    
            
        panorama_image = imload(panorama_path)
        
        db.set_variable("panorama_shape",panorama_image.shape)
        
        if panorama_image.shape[2] != 3:
            print(f"Image has more than 3 channels. Quitting...")
            exit()
            
        db.set_variable("panorama_image", panorama_image)
        
        H, W = panorama_image.shape[:2]
        
        db.set_variable("W",W)
        db.set_variable("H",H) 
        
        db.set_variable("panorama_path",panorama_path)
        
        # imshow(panorama_image, "Panorama Image") 
        
        panorama_name = ".".join(os.path.basename(panorama_path).split(".")[:-1])
        
        db.set_variable("panorama_name",panorama_name)
        
    pick_file(cm.database)
        
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
        
        db.set_variable(cell_mask_name, cell_mask)
        

    @cm.checkpoint
    def get_alpha_shapes(db: Database):

        # no need to call seq_cell_mask.next, as durign and following this step, the name "cell_layer_mask" is used

        pr=Progress("get_alpha_shapes")

        with pr.track("read database"):
            cell_mask = db.get_variable(seq_cell_mask.current())

        H, W = cell_mask.shape

        ds = WRAP_ISLANDS.DS

        cell_mask = IP.BooleanImageOps.scale_by(cell_mask, 1/ds)

        with pr.track("get nonzero locations"):
            pixel_locations_xy = np.flip(np.transpose(np.nonzero(cell_mask)),axis=1)

        with pr.track("get alphashapes"):
            polys = IP.run_alphashape_and_get_polygons(pixel_locations_xy,WRAP_ISLANDS.ALPHA)

        with pr.track("write database"):

            db.set_variable("island_polys",polys) #@TODO: double check that variables are indirect by default

        cell_layer_mask = np.zeros_like(cell_mask)

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
        
        HH, WW = wing_mask.shape

        boundary = spine_analysis.get_boundary(wing_mask)

        naive_centerline = spine_analysis.get_naive_centerline(boundary)

        true_spine = spine_analysis.convert_naive_spine_to_real_spine(boundary,naive_centerline)

        boundary = (FIND_SPINE.DS*boundary.astype(float)).astype(int)

        naive_centerline = (FIND_SPINE.DS*naive_centerline.astype(float)).astype(int)

        true_spine = (FIND_SPINE.DS*true_spine.astype(float)).astype(int)

        proxy.set_variable("wing_boundaries",boundary)

        proxy.set_variable("wing_naive_centerlines",naive_centerline)

        proxy.set_variable("true_spines",true_spine)

        curve_image = np.full((HH,WW,3),255,np.uint8)

        pallete = IP.Pallete()
        
        with pr.track("Visualizing boundaries and spines..."):

            curve_image = IP.Image.draw_curve_on(curve_image,boundary.astype(float)/FIND_SPINE.DS,pallete.nextColor(),2,closed=True)

            # curve_image = IP.Image.draw_curve_on(curve_image,naive_centerline.astype(float)/FIND_SPINE.DS,pallete.nextColor(),2,closed=False)

            curve_image = IP.Image.draw_curve_on(curve_image,true_spine.astype(float)/FIND_SPINE.DS,pallete.nextColor(),2,closed=False)

        curve_image = cv2.resize(curve_image,dsize=(W,H),interpolation=cv2.INTER_NEAREST)

        imshow(curve_image,f"Boundary and Centerline -- Wing {proxy.index()+1} of {proxy.size()}")

    cm.register_mapped_checkpoint("cell_layer_wings",get_boundaries_and_spines)

    def cut_mask_into_groups(db: Database, proxy: MapProxy):
        
        pr = Progress("cut_mask_into_groups")
        
        panorama_image = db.get_variable("panorama_image")

        H, W = db.get_variable("panorama_shape")[:2]

        wing_mask = proxy.get_variable("cell_layer_wings")

        wing_mask = IP.BooleanImageOps.scale_by(wing_mask,1/CUT_MASK_INTO_GROUPS.DS)

        nonzero_locations_xy = IP.BooleanImageOps.where2D(wing_mask)

        print("wing_boundaries: "+str(db.get_variable("wing_boundaries")))

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
            

            iA = segment_offs+group_number*segments_per_group
            iB = segment_offs+(group_number+1)*segments_per_group

            print(group_number, num_groups, iA,iB,len(group_curve), min(iB,len(group_curve)-1)-min(iA,len(group_curve)-1))

            for nonzero_location in nonzero_locations_xy:
                d, i = spine_tree.query(nonzero_location,1)
                x,y = nonzero_location
                if i >= iA and i <= iB:
                    group_mask[y,x] = True

            group_mask = IP.BooleanImageOps.resize_to(group_mask,W,H)
            
            panorama_name = db.get_variable("panorama_name")
            
            with pr.track(f"Creating manual annotation data for wing {proxy.index()} group {group_number}..."):
                image_for_manual_annotation = IP.BooleanImageOps.select_image_region_by_mask(panorama_image,group_mask,(255,255,255))
                imsave(image_for_manual_annotation,f"output/images_for_manual_annotation/{panorama_name}/{proxy.index()}.{group_number}.png")
                
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
            
            group_counts.append(np.max(watershed_labels)+1)

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
        
        print(group_counts)
        
        group_masks = db.get_variable("group_masks")
        
        edge_plot_grouping = np.zeros_like(cell_layer_mask).astype(int) -1
        
        vis_watershed_labels = db.get_variable("vis_watershed_labels")
        
        print(vis_watershed_labels)
        
        all_watershed_labels = np.zeros((H,W),int)-1
        
        panorama_name = db.get_variable("panorama_name")
        
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
                
                plot = IP.Draw.draw_annotation(plot,f"{int(group_count)+1}",annotation_position[0],annotation_position[1],5,(255,0,0),4,0.75,4)
        
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
        
        imsave(plot,f"output/{panorama_name}/Group Counts Plot")  
        
        with open(f"output/{panorama_name}/group_counts.tsv","w") as fl:
            
            fl.write("\tWing\tGroup\t# Nuclei")
            
            for wing in range(num_wings):
                
                num_groups = db.get_variable("num_groups")[wing]
                
                wing_group_counts = group_counts[wing]
            
                for group_number in range(num_groups):
                    
                    group_count = wing_group_counts[group_number]
                    
                    fl.write(f"{int(wing)+1}\t{int(group_number)+1}\t{int(group_count)}\n")
            
    cm.menu()