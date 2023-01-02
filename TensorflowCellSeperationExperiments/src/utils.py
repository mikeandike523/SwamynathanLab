import os
import shutil
import math
import numpy as np
import dill
import argparse
from collections import defaultdict
import re
from tkinter import filedialog
from tkinter import Tk
import time
import functools

def npAndMany(*args):
    return functools.reduce(np.logical_and,args)

def npOrMany(*args):
    return functools.reduce(np.logical_or, args)


FS_POLL_TIME = 0.100 #s

def userappdatafolder():
    path = os.getenv('USERAPPDATA')
    path = path.replace('\\','/')
    path = path.rstrip('/')
    return path

def blocking_unlink(path):
    os.unlink(path)
    while os.path.isfile(path):
        time.sleep(FS_POLL_TIME)

def blocking_rmtree(path):
    shutil.rmtree(path)
    while os.path.isdir(path):
        time.sleep(FS_POLL_TIME)

def askopenfilename():
    tk = Tk()
    tk.withdraw()
    return filedialog.askopenfilename()

def rescale_array(arr):
    if np.count_nonzero(arr) == 0:
        return np.zeros_like(arr)
    if np.all(arr==np.max(arr)):
        return np.ones_like(arr)
    return (arr.copy()-np.min(arr))/(np.max(arr) - np.min(arr))

def init_folder(path, clear=True):
    if not os.path.isdir(path):
        try:
            os.makedirs(path,exist_ok=True)
        except:
            os.mkdir(path)
    if clear:
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isfile(item_path):
                blocking_unlink(item_path)
            if os.path.isdir(item_path):
                blocking_rmtree(item_path)

def flip(arr_1d):
    return np.flip(arr_1d)

def opt_min(a,b):
    if a is None and b is None:
        raise ValueError("Either a or b should be non-null")
    if a is None:
        return b
    if b is None:
        return a
    return min(a,b)

def opt_max(a,b):
    if a is None and b is None:
        raise ValueError("Either a or b should be non-null")
    if a is None:
        return b
    if b is None:
        return a
    return max(a,b)

# --- Adapted from https://stackoverflow.com/a/24439444/5166365
import inspect
import os
import time
last_time = None
def dprint(message, show_elapsed_time = False):
    global last_time
    current_time = time.time()
    elapsed_time = 0 if last_time is None else current_time - last_time
    last_time = current_time
    caller = inspect.getframeinfo(inspect.stack()[1][0])
    print(f"| {'{:.2f}s | '.format(elapsed_time) if show_elapsed_time else ''}{os.path.relpath(caller.filename)} line {caller.lineno}: {message}")
# ---

def coefficient_of_determination(ground_truth, estimated):

    "R^2 value"

    y = np.array(ground_truth,float)
    
    ybar = np.mean(y)
    
    yhat = np.array(estimated,float)

    SST = np.sum(np.power(y-ybar,2))

    SSE = np.sum(np.power(yhat-y,2))

    return 1.0-SSE/SST

def bin_centers(bin_edges):

    return np.mean([
        np.array(bin_edges[1:]),
        np.array(bin_edges[:-1]),
    ],axis=0)

def signum(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0

def angleOfVector(v):
    return math.atan2(v[1], v[0])

def principalAngleOfVector(v):
    theta = angleOfVector(v)
    if theta < 0:
        theta += 2 * math.pi
    return theta

def signedChangeInHeading(src, dst):
    theta_src = principalAngleOfVector(src)
    theta_dst = principalAngleOfVector(dst)
    initial_sign = signum(theta_dst-theta_src)
    initial_guess_magnitude = abs(theta_dst-theta_src)
    alternate_guess_magnitude = 2*math.pi-abs(theta_dst-theta_src)
    ret_magnitude = initial_guess_magnitude
    ret_sign = initial_sign
    if alternate_guess_magnitude < ret_magnitude:
        ret_sign = -initial_sign
        ret_magnitude = alternate_guess_magnitude
    return ret_sign * ret_magnitude

# Courtesy of Yves Daost on stackoverflow
# https://stackoverflow.com/a/74238178/5166365
def fast_signedChangeInHeading(src,dst):
    return math.atan2(src[0]*dst[1]-src[1]*dst[0],src[0]*dst[0]+src[1]*dst[1])

def estimate_relative_curvature(contour, target_idx, extent):
    """ Estimate the curvature of a closed contour at a given point using arbitrary units
    """

    fwd_total_heading_change = 0
    for didx in range(-extent,extent+1):
        i1 = target_idx
        i2 = (target_idx+didx) % len(contour)
        src = contour[i1]
        dst = contour[i2]
        fwd_total_heading_change += fast_signedChangeInHeading(src,dst)
    fwd_mean_heading_change =  fwd_total_heading_change / (2*extent+1)

    bwd_total_heading_change = 0
    for didx in range(-extent,extent+1):
        i1 = target_idx
        i2 = (target_idx-didx) % len(contour)
        src = contour[i1]
        dst = contour[i2]
        bwd_total_heading_change += fast_signedChangeInHeading(src,dst)
    bwd_mean_heading_change =  bwd_total_heading_change / (2*extent+1)

    return 0.5 * fwd_mean_heading_change + 0.5 * bwd_mean_heading_change

def scalar_project(base,target):
    return np.dot(target,base) / np.linalg.norm(base)

def format_path(path,strip_whitespace=True):

    path = path.replace('\r\n','\n')

    if strip_whitespace:
        path = path.strip(' ')
        path = path.strip('\t')
        # This should not be necessary
        path = path.strip('\n')

    path = path.replace('\\','/')
    path = re.sub("/+","/",path)
    path = path.rstrip('/')
    return path

class Math:

    @classmethod
    def signum(cls,value):
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

class Geometry:

    class CURVE_ORIENTATION: # pythonic way of making an enum

        CW = 1
        CCW = -1
        COLINEAR = 0

    @classmethod
    def get_window_locations_covering_image(cls,image_W,image_H,window_W,window_H, overlap_fraction_x = 0.0, overlap_fraction_y = 0.0):

        stride_x = int(window_W-overlap_fraction_x*image_W)

        stride_y = int(window_H-overlap_fraction_y*image_H)

        covered = np.zeros((window_H,window_W),dtype=bool)

        x = 0

        y = 0

        locations = []

        while np.any(np.logical_not(covered)):
            
            locations.append((x,y))

            # Taking advantage of the fact that numpy arrays won't error when setting a rectangular window that may fall partway outside the bounds of the array

            covered[y:y+window_H, x:x+window_W] = True

            x += stride_x

            if x >= image_W:

                x = 0

                y += stride_y

            if y >= image_H:

                break

        return locations


    # --- Algorithm courtesy of https://en.wikipedia.org/wiki/Curve_orientation#:~:text=External%20links-,Orientation%20of%20a%20simple%20polygon,-%5Bedit%5D
    @classmethod
    def check_orientation(cls,curve):

        def check_direction_at_bend(bend_point_A, bend_point_B, bend_point_C):

            Ax, Ay = bend_point_A
            Bx, By = bend_point_B
            Cx, Cy = bend_point_C

            bend_matrix = np.array([
                [1,Ax, Ay],
                [1,Bx, By],
                [1,Cx, Cy]
            ],float)

            return Math.signum(np.linalg.det(bend_matrix))

        total_direction = 0

        print(curve)

        for i in range(len(curve)):
            
            iA = (i-1) % len(curve)
            iB = (i) % len(curve)
            iC = (i+1) % len(curve)

            total_direction +=check_direction_at_bend(curve[iA],curve[iB],curve[iC])

        if total_direction > 0:
            return Geometry.CURVE_ORIENTATION.CW

        if total_direction < 0:
            return Geometry.CURVE_ORIENTATION.CCW

        # Unlikely to ever happend
        return Geometry.CURVE_ORIENTATION.COLINEAR


    # ---

    @classmethod
    def normalizeVector(cls, A,epsilon=1e-6):
        M = np.linalg.norm(A)
        if M < epsilon:
            raise ValueError(f"Vector {A} is too small to be normalized.")
        return 1/M * A

    @classmethod
    def scalarProjection(cls,A,B, epsilon=1e-6):
        return np.dot(A,Geometry.normalizeVector(B.copy(),epsilon))
            