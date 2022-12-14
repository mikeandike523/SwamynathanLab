import sys

import cv2

import utils as U
import image_processing as IP


def perform_watershed(image_section,mask_section, grouping_section):
    
    seed_grouping = grouping_section

    LAB_image = cv2.cvtColor(image_section, cv2.COLOR_RGB2LAB)

    L_channel = U.rescale_array(LAB_image[:,:,0])

    watershed_channel = IP.greyscale_plot_to_color_image(L_channel)

    seed_grouping_watershed = IP.Grouping.watershed(mask_section, seed_grouping,watershed_channel)

    return seed_grouping_watershed