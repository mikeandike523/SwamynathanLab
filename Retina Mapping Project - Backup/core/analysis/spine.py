import copy

import numpy as np
import cv2
from scipy.spatial import KDTree

import image_processing as IP
import utils as U

# applicaiton parameters
RESAMPLE_PIXELS = 50
INSET_PIXELS = 200
MIN_PIXELS_BETWEEN_CORNERS = 100
FIND_CORNERS_K = 40 # Number of nearest neighbors to check when finding ROI corners

def get_naive_centerline(boundary):
    """! Get the 'naive' centerline of a region enclosed by `boundary` 
    
    Procedure:

        Calculate the region enclosed by the boundary --> `mask`

        If the number of points in the boundary is even, add a point as the midpoint of the first two points

        For each point in the boundary as `startpoint`:

            pair up all of the remaining points starting at both sides of the starting point, and calculate the "test centerline" as the sequence midpoints of these pairs

            Rasterize the test centerline with thickness of 1

            record the number of pixels in the intersection between the raster and mask

        Choose the starting point that maximizes the number of pixels in the intersection

    """

    mask = IP.Curve.curve_to_mask(boundary)

    if len(boundary) % 2 == 0:
        A = np.array(boundary[0],float)
        B = np.array(boundary[1],float)
        M = 0.5 * A + 0.5 * B
        boundary = np.insert(boundary,1,M,axis=0)

    def get_boundary_point(i):
        return boundary[i%len(boundary)]

    wing_length = len(boundary)//2

    def get_spine_at_starting_point(starting_point):
        spine_points = []
        
        for i in range(wing_length):
            A = get_boundary_point(starting_point-i)
            B = get_boundary_point(starting_point+i)
            M = 0.5*A + 0.5 *B

            spine_points.append(M)
            
        spine_points = np.array(spine_points,int)

        return spine_points

    starting_point_metrics = []

    for starting_point in range(len(boundary)):
        
        print(f"Starting point {starting_point+1} of {len(boundary)}")

        spine_points = get_spine_at_starting_point(starting_point)

        line_mask = IP.BooleanImageOps.draw_curve_on(np.zeros_like(mask),spine_points)

        intersection_size = np.count_nonzero(np.logical_and(mask, line_mask))

        starting_point_metrics.append(intersection_size)

    best_starting_point = np.argmax(starting_point_metrics)

    return get_spine_at_starting_point(best_starting_point)

    
        


def get_boundary(mask):

    # Ensure that there is only one connected body, using connectivity = 8
    
    grouping = IP.BooleanImageOps.connected_components(mask,8)

    if np.max(grouping) > 0:
        raise ValueError("The provided mask has more than one connected body.")

    # Get the boundary of the pixel island, ensure
    # The image_processing library contains a similar function to the following, but is more limited in detecting edge-cases
    mask_image = IP.greyscale_plot_to_color_image(mask)[:,:,0]
    contours, hierarchy = cv2.findContours(mask_image,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    if len(contours) != 1: 
        raise ValueError("There was more than one contour. The boundary may be too rough or the pixel island may contain a hole.")

    boundary = contours[0].squeeze()

    #@TODO: Re-Read documentation, unclear if contour is reported in x,y or r,c. 
    # The documentation mentions that the function returns a vector of points, whose constructor is x, y
    # Nevertheless, that doesn't necessarily mean the point constructor wasn't loaded with x=r, y=c
    # For now, assume x,y
    
    return IP.Curve.resample_by_segment_length(boundary,RESAMPLE_PIXELS,True)

def convert_naive_spine_to_real_spine(boundary,naive_spine):

    if len(boundary) % 2 != 0:
        A = np.array(boundary[0],float)
        B = np.array(boundary[1],float)
        M = (0.5 * A + 0.5 * B).astype(int)
        boundary = np.insert(boundary,1,M,axis=0)
        
    # Ensure that the boundary is counter-clockwise

    if U.Geometry.check_orientation(boundary) != U.Geometry.CURVE_ORIENTATION.CCW:
        boundary = np.flip(boundary,axis=0)

    inset_vertex_count = int(np.ceil(INSET_PIXELS/RESAMPLE_PIXELS))

    min_idxs_between_corners = int(np.ceil(MIN_PIXELS_BETWEEN_CORNERS/RESAMPLE_PIXELS))

    inset_spine_begin_idx = inset_vertex_count

    inset_spine_end_idx = len(naive_spine) -1 - inset_vertex_count # (inclusive)

    boundary_vertex_tree = KDTree(boundary)

    def travel_ccw(idx,delta):
        return (idx + delta) % len(boundary)

    def travel_cw(idx,delta):
        return (idx - delta) % len(boundary)

    def get_absolute_idx(idx):
        return idx % len(boundary)

    def travel_in_direction(idx,direction):
        return (idx+direction) % len(boundary)

    def get_longest_distance_between_idxs(idxA, idxB):

        idxA = get_absolute_idx(idxA)
        idxB = get_absolute_idx(idxB)

        dist1 = abs(idxB - idxA)
        dist2 = len(boundary) - dist1
        return max(dist1,dist2)
        
    def get_shortest_distance_between_idxs(idxA, idxB):

        idxA = get_absolute_idx(idxA)
        idxB = get_absolute_idx(idxB)

        dist1 = abs(idxB - idxA)
        dist2 = len(boundary) - dist1
        return min(dist1,dist2)

    def get_direction_from_idxA_to_idxB(idxA,idxB):
        
        idxA = get_absolute_idx(idxA)
        idxB = get_absolute_idx(idxB)

        delta_ccw = idxB-idxA
      
        short = get_shortest_distance_between_idxs(idxA,idxB)
        long = get_longest_distance_between_idxs(idxA,idxB)

        if abs(delta_ccw) == short:
            if delta_ccw < 0:
                return U.Geometry.CURVE_ORIENTATION.CW
            return U.Geometry.CURVE_ORIENTATION.CCW
        else:
            if delta_ccw < 0:
                return U.Geometry.CURVE_ORIENTATION.CCW
            return U.Geometry.CURVE_ORIENTATION.CW


        
    def get_corner_pair(naive_spine_idx):
        
        dd, ii = boundary_vertex_tree.query(naive_spine[naive_spine_idx],FIND_CORNERS_K)

        # Ensure that values are sorted and are of the right type. Probably not necessary. Need to read the docs on KDTree.query()
        sortorder = np.argsort(dd)
        dd = np.array(dd,float)
        ii = np.array(ii,int)
        dd = dd[sortorder]
        ii = ii[sortorder]

        idx_A = ii[0]

        idx_B = None

        for i in ii[1:]:
            idx_dist = get_shortest_distance_between_idxs(idx_A,i)
            if idx_dist >= min_idxs_between_corners:
                idx_B = i
                break

        if idx_B is None:
            raise ValueError("Could not find an appropriate second corner with the supplied parameters.")

        return (idx_A, idx_B)

    corner_pair_tip = get_corner_pair(inset_spine_begin_idx)

    corner_pair_tail = get_corner_pair(inset_spine_end_idx)

    def find_meeting_points(idxA, idxB):

        direction_AB = get_direction_from_idxA_to_idxB(idxA,idxB)
        direction_BA = get_direction_from_idxA_to_idxB(idxB,idxA)
        
        print(direction_AB, direction_BA)

        whose_turn = 0

        while idxA != idxB:

            if whose_turn % 2 == 0:
                idxA = travel_in_direction(idxA,direction_AB)
            else:
                idxB = travel_in_direction(idxB,direction_BA)

            whose_turn += 1
    
        return idxA

    true_spine_tip_idx = find_meeting_points(*corner_pair_tip)

    true_spine_tail_idx = find_meeting_points(*corner_pair_tail) 

    while get_shortest_distance_between_idxs(true_spine_tail_idx,true_spine_tip_idx) != get_longest_distance_between_idxs(true_spine_tail_idx,true_spine_tip_idx):
        direction_tail_tip = get_direction_from_idxA_to_idxB(true_spine_tail_idx,true_spine_tip_idx)
        true_spine_tail_idx = travel_in_direction(true_spine_tail_idx,direction_tail_tip)

    spine_points = []

    num_pairs = int((len(boundary)-2)/2)

    p1 = travel_cw(true_spine_tip_idx,1)
    p2 = travel_ccw(true_spine_tip_idx,1)

    for _ in range(num_pairs):
    
        A = np.array(boundary[p1],float)
        B = np.array(boundary[p2],float)
        M = (0.5 * A + 0.5 * B).astype(int)

        spine_points.append(M)

        p1 = travel_cw(p1,1)
        p2 = travel_ccw(p2,1)

    return np.array(spine_points,float)