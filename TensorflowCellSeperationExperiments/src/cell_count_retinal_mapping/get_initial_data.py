
def exports():

    import numpy as np
    import cv2
    import PIL.Image
    import shapely.geometry
    import skimage.draw
    from scipy.spatial import KDTree
    from types import SimpleNamespace
    import pickle

    import env_setup
    env_setup.env_setup()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset-state',action='store_true')

    args = parser.parse_args()

    import os

    pwd = os.getcwd()

    os.chdir("../../src")

    if not args.reset_state and os.path.isfile("temp/state.pkl"):
        with open("temp/state.pkl","rb") as fl:
            return pickle.load(fl)

    import image_processing as IP
    import utils as U

    SPINE_DOWNSAMPLING = 4
    RESAMPLE_NUM_POINTS = 300
    TESTLINE_THICKNESS = 3
    INSET = 25
    REVERSE_INSET_C1_C3 = 0
    REVERSE_INSET_C2_C4 = 0
    CORNER_MIN_SPACING = 10
    CORNER_QUERY_COUNT = 20
    FINAL_ADJUSTMENT_DISTANCE = 1 # number of indices forward/backward to test to minimize corner distance 
    CORNER_MEET_DISTANCE_C1_C3 = 5
    CORNER_MEET_DISTANCE_C2_C4 = 5

    mask_filepath = "../assets/dev/current_best_mask.png"

    mask_image = np.asarray(PIL.Image.open(mask_filepath))

    original_mask_image = mask_image.copy()

    fullscaleH, fullscaleW = mask_image.shape[:2]

    mask_image = IP.scale_image_nearest(mask_image,1/SPINE_DOWNSAMPLING)

    mask_image = np.expand_dims(mask_image[:,:,0],axis=2)

    plots = []

    def add_plot(plot):
        if plot.ndim == 2:
            plots.append(IP.greyscale_plot_to_color_image(plot.copy()))
        else:
            plots.append(plot.copy())
        

    add_plot(np.dstack((mask_image.squeeze(),)*3))

    contours,_ = cv2.findContours(mask_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    H, W = mask_image.shape[:2]

    contour = contours[np.argmax([len(contour.squeeze()) for contour in contours])].squeeze()

    """Find centerline"""

    def resample_closed_contour(closed_contour, num_segments):
        resampled_contour = []
        polygon = shapely.geometry.LinearRing(list(closed_contour))
        for i in range(num_segments):
            new_point = np.array(polygon.interpolate(i/num_segments,normalized=True),float)
            resampled_contour.append(new_point.astype(int))
        resampled_contour = np.array(resampled_contour)
        return resampled_contour

    cimg = np.zeros((H,W,3),dtype=np.uint8)

    for i in range(len(contour)):
        p1 = contour[i]
        p2 = contour[(i+1)%len(contour)]
        cimg = cv2.line(cimg,p1.astype(int),p2.astype(int),(255,255,255),2)

    contour = resample_closed_contour(contour,RESAMPLE_NUM_POINTS)

    for i in range(len(contour)):
        p1 = contour[i]
        p2 = contour[(i+1)%len(contour)]
        cimg = cv2.line(cimg,p1.astype(int),p2.astype(int),(255,0,0),2)

    def get_spine(contour):

        contour = np.array(contour)

        if len(contour) % 2 == 1:
            A = contour[0].astype(float)
            B = contour[1].astype(float)
            M = 0.5*A+0.5*B
            contour.insert(1,M.astype(int))
        
        num_points = len(contour)

        measurements = []

        mask1 = skimage.draw.polygon2mask((H,W), np.flip(contour,axis=1))

        IP.imshow(IP.greyscale_plot_to_color_image(mask1),"test")

        for start_point in range(num_points):

            print(f"Measuring start-point {start_point+1} of {num_points}...")
            
            testline = []
            total_error = 0
            total_len = 0

            distances = []

            num_pairs = num_points // 2
            for i in range(num_pairs):
                ia = (start_point + i) % num_points
                ib = (start_point + num_points-1-i) % num_points
                A = contour[ia].astype(float)
                B = contour[ib].astype(float)
                M =0.5*A + 0.5 * B
                distances.append(np.linalg.norm(A-M))
                distances.append(np.linalg.norm(B-M))
                testline.append(M.astype(int))
                if len(testline) > 0:
                    total_len += np.linalg.norm(testline[-1].astype(float)-M)
            
            total_error = np.std(distances)

            mask2_image = np.zeros((H,W,3),dtype=np.uint8)

            for i in range(len(testline)-1):
                mask2_image = cv2.line(mask2_image,testline[i],testline[i+1],(255,255,255),TESTLINE_THICKNESS)

            mask2 = mask2_image[:,:,0] > 0

            IP.imshow(IP.greyscale_plot_to_color_image(mask2),"test2")

            miss_fraction = np.count_nonzero(np.logical_and(mask2,np.logical_not(mask1))) / np.count_nonzero(mask2)

            measurements.append(miss_fraction+total_error)

        start_point = np.argmin(measurements)

        spine = []

        num_pairs = num_points // 2
        for i in range(num_pairs):
            ia = (start_point + i) % num_points
            ib = (start_point + num_points-1-i) % num_points
            A = contour[ia].astype(float)
            B = contour[ib].astype(float)
            M = 0.5 *A + 0.5 * B 
            spine.append(M.astype(int))

        return np.array(spine,float)

    spine = get_spine(contour)

    spine = spine[INSET:(len(spine)-INSET)]

    tree = KDTree(contour.astype(float))

    tip = spine[0].astype(float)
    tail = spine[-1].astype(float)

    def index_min_distance(i1, i2):
        return np.min([abs(i2-i1),len(contour) - abs(i2-i1)])

    def find_corners_near_point(pt):
        dd, ii = tree.query(pt,CORNER_QUERY_COUNT)
        
        sortorder = np.argsort(dd) #is the result of tree.query already sorted?

        dd = dd[sortorder]
        ii = ii[sortorder]

        first_corner_index = ii[0]    
        second_corner_index = None
        for test_corner_index in ii[1:]:
            if index_min_distance(test_corner_index, first_corner_index) >= CORNER_MIN_SPACING:
                second_corner_index = test_corner_index
                break
        if second_corner_index is None:
            raise Exception(f"Could not find second corner near point {pt}.")
        return first_corner_index, second_corner_index
        
    icorner1, icorner2 = find_corners_near_point(tip)
    icorner3, icorner4 = find_corners_near_point(tail)

    for i in range(len(spine)-1):
        p1 = spine[i]
        p2 = spine[i+1]
        cimg = cv2.line(cimg,p1.astype(int),p2.astype(int),(0,255,0),2)

    add_plot(cimg)

    figure = IP.ImageStrip(*plots)
    IP.imshow(figure.getImagePixels(),"downscaled")

    contour=np.array(contour,float)
    contour=SPINE_DOWNSAMPLING * contour

    spine = np.array(spine,float)
    spine = SPINE_DOWNSAMPLING * spine

    fullscale_mask_image = original_mask_image
    fullscale_mask = fullscale_mask_image[:,:,0] > 0

    vis_image = np.zeros_like(fullscale_mask_image)

    vis_image[fullscale_mask,...] = (255,255,255)

    cA, cB, cC, cD = icorner1, icorner2, icorner3, icorner4
    vis_image = cv2.circle(vis_image, contour[cA].astype(int), 10,(0,255,255),-1)
    vis_image = cv2.circle(vis_image, contour[cB].astype(int), 10,(0,255,255),-1)
    vis_image = cv2.circle(vis_image, contour[cC].astype(int), 10,(0,255,255),-1)
    vis_image = cv2.circle(vis_image, contour[cD].astype(int), 10,(0,255,255),-1)

    corners = [cA, cB, cC, cD]

    CA = cA
    remaining = [cB, cC, cD]
    CB = remaining[np.argmin([np.linalg.norm(contour[CA]-contour[c]) for c in remaining])]
    print(CB)
    remaining.remove(CB)
    CC = remaining[0]
    CD = remaining[1]

    #if CD is closer (in index) to CB, then swap CC and CD
    if index_min_distance(CD,CB) < index_min_distance(CC,CB):
        CC, CD = CD, CC

    def delta_to(A, B):
        delta = int((B-A)/abs(B-A))
        travel = abs(B-A)
        print(index_min_distance(A,B),travel)
        if index_min_distance(A,B) != travel:
            delta *= -1
        return delta

    # record the long sides
    long_sides = [(CB,CC),(CA, CD)]

    print(long_sides)

    (C1, C2), (C3, C4) = long_sides

    init_C1 = C1
    init_C2 = C2
    init_C3 = C3
    init_C4 = C4

    delta_1 = delta_to(init_C1,init_C3)
    delta_2 = delta_to(init_C2, init_C4)
    delta_3 = delta_to(init_C3,init_C1)
    delta_4 = delta_to(init_C4,init_C2)

    """Adjust corner pair C1+C3"""
    post_adjustment_metrics = []
    for d1 in range(-FINAL_ADJUSTMENT_DISTANCE,FINAL_ADJUSTMENT_DISTANCE+1,1):
        for d2 in range(-FINAL_ADJUSTMENT_DISTANCE,FINAL_ADJUSTMENT_DISTANCE+1,1):
            new_C1 = (C1 + d1) % len(contour)
            new_C3 = (C3 + d2) % len(contour)
            dist = np.linalg.norm(contour[new_C1]-contour[new_C3])
            post_adjustment_metrics.append(((new_C1,new_C3),dist))
    post_adjustment_metrics = sorted(post_adjustment_metrics, key=lambda entry: entry[1])
    C1, C3 = post_adjustment_metrics[0][0]

    """Adjust corner pair C2+C4"""
    post_adjustment_metrics = []
    for d1 in range(-FINAL_ADJUSTMENT_DISTANCE,FINAL_ADJUSTMENT_DISTANCE+1,1):
        for d2 in range(-FINAL_ADJUSTMENT_DISTANCE,FINAL_ADJUSTMENT_DISTANCE+1,1):
            new_C2 = (C2 + d1) % len(contour)
            new_C4 = (C4 + d2) % len(contour)
            dist = np.linalg.norm(contour[new_C2]-contour[new_C4])
            post_adjustment_metrics.append(((new_C2,new_C4),dist))
    post_adjustment_metrics = sorted(post_adjustment_metrics, key=lambda entry: entry[1])
    C2, C4 = post_adjustment_metrics[0][0]

    """Walk until the corners are close to the edge of the roi"""
    counter = 0
    while index_min_distance(C1,C3) > CORNER_MEET_DISTANCE_C1_C3:
        if counter % 2 == 0:
            C1 = (C1 + delta_1) % len(contour)
        else:
            C3 = (C3 + delta_3) % len(contour)
        counter += 1
    while index_min_distance(C2,C4) > CORNER_MEET_DISTANCE_C2_C4:
        if counter %2 ==0:
            C2 = (C2 + delta_2) % len(contour)
        else:
            C4 = (C4 + delta_4) % len(contour)
        counter += 1

    """Adjust corner pair C1+C3"""
    post_adjustment_metrics = []
    for d1 in range(-FINAL_ADJUSTMENT_DISTANCE,FINAL_ADJUSTMENT_DISTANCE+1,1):
        for d2 in range(-FINAL_ADJUSTMENT_DISTANCE,FINAL_ADJUSTMENT_DISTANCE+1,1):
            new_C1 = (C1 + d1) % len(contour)
            new_C3 = (C3 + d2) % len(contour)
            dist = np.linalg.norm(contour[new_C1]-contour[new_C3])
            post_adjustment_metrics.append(((new_C1,new_C3),dist))
    post_adjustment_metrics = sorted(post_adjustment_metrics, key=lambda entry: entry[1])
    C1, C3 = post_adjustment_metrics[0][0]

    """Adjust corner pair C2+C4"""
    post_adjustment_metrics = []
    for d1 in range(-FINAL_ADJUSTMENT_DISTANCE,FINAL_ADJUSTMENT_DISTANCE+1,1):
        for d2 in range(-FINAL_ADJUSTMENT_DISTANCE,FINAL_ADJUSTMENT_DISTANCE+1,1):
            new_C2 = (C2 + d1) % len(contour)
            new_C4 = (C4 + d2) % len(contour)
            dist = np.linalg.norm(contour[new_C2]-contour[new_C4])
            post_adjustment_metrics.append(((new_C2,new_C4),dist))
    post_adjustment_metrics = sorted(post_adjustment_metrics, key=lambda entry: entry[1])
    C2, C4 = post_adjustment_metrics[0][0]


    vis_image = cv2.circle(vis_image, contour[C1].astype(int),20,(255,0,0), 5)
    vis_image = cv2.circle(vis_image, contour[C2].astype(int),20,(0,255,0), 5)
    vis_image = cv2.circle(vis_image, contour[C3].astype(int),20,(0,0,255), 5)
    vis_image = cv2.circle(vis_image, contour[C4].astype(int),20,(255,255,255), 5)


    long_side_A_idxs = []
    long_side_B_idxs = []

    pointer = C1
    delta = delta_to(init_C1,init_C2)
    print(delta)
    while pointer != C2:
        long_side_A_idxs.append(pointer)
        pointer = (pointer + delta) % len(contour)

    pointer = C3
    delta = delta_to(init_C3,init_C4)
    print(delta)
    while pointer != C4:
        long_side_B_idxs.append(pointer)
        pointer = (pointer + delta) % len(contour)

    for i in range(len(contour)-1):
        vis_image = cv2.line(vis_image,contour[i].astype(int),
                contour[i+1].astype(int),(128,128,128),12)

    for i in range(len(long_side_A_idxs)-1):
        vis_image = cv2.line(vis_image,contour[long_side_A_idxs[i]].astype(int),
                contour[long_side_A_idxs[i+1]].astype(int),(255,255,0),12)

    for i in range(len(long_side_B_idxs)-1):
        vis_image = cv2.line(vis_image,contour[long_side_B_idxs[i]].astype(int),
                contour[long_side_B_idxs[i+1]].astype(int),(0,255,255),12)

    long_side_A = contour[long_side_A_idxs]
    long_side_B = contour[long_side_B_idxs]

    length_A = IP.Curve.measure_length(long_side_A)
    length_B = IP.Curve.measure_length(long_side_B)

    small_curve = long_side_A
    large_curve = long_side_B

    if length_A > length_B:
        small_curve, large_curve = large_curve, small_curve

    large_curve_tree = KDTree(large_curve)

    spine = []

    for i in range(len(small_curve)):
        A = small_curve[i].astype(float)
        _, i2 = large_curve_tree.query(A,1)
        B = large_curve[i2].astype(float)
        M = 0.5 * A + 0.5 * B
        spine.append(M.astype(int))

    spine = np.array(spine,int)
        
    for i in range(len(spine)-1):
        vis_image = cv2.line(vis_image,spine[i],
                spine[i+1],(100,200,50),12)


    # fs = "full scale"

    state = SimpleNamespace(**{
        "fs_mask":original_mask_image[:,:,0] > 0,
        "fs_boundary": np.array(contour),
        "fs_spine":np.array(spine),
        "fs_image":np.asarray(PIL.Image.open("../assets/golden_retina_map.png"))
    })

    with open("temp/state.pkl","wb") as fl:
        pickle.dump(state,fl)

    return state