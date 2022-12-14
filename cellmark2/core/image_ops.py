import cv2

def resize(pixels, target_w, target_h, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(pixels, (target_w, target_h), interpolation=interpolation)

def scale(pixels, scale_factor, interpolation=cv2.INTER_LINEAR):
    target_w = int(pixels.shape[1]*scale_factor)
    target_h = int(pixels.shape[0]*scale_factor)
    return resize(pixels, target_w, target_h, interpolation)