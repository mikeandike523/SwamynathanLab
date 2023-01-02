import os

import numpy as np

import io3.io3 as rustcore

os.system("color")

from termcolor import colored

class Image:
    
    def __init__(self, w, h, nchannels):
        self.w = w
        self.h = h
        self.nchannels = nchannels
        self.data = list(np.zeros((w*h*nchannels,),dtype=float))
        self.rimg = rustcore.Image(self.w, self.h, self.nchannels)
        self.dtype = None

        if ((os.getenv('IO3_TIPS') or "").strip(' ').lower() != 'off'):
            print(colored("class io3.pycore.Image: Note, the only function that automatically calls `Image.swap_back()` is `Image.to_array()`. Remember to manually call `Image.swap_back()` when needed to syncronize python-side and rust-side data. Set environment variable `IO3_TIPS=off` to disable warnings.","yellow"))
        
    def __set_data(self, data):
        self.data = data
        self.rimg.set_data(self.data)    
    
    @classmethod
    def from_array(cls,arr):
        h, w , nchannels = arr.shape
        result = cls(w,h,nchannels)
        result.__set_data(list(np.ravel(arr.astype(float))))
        result.dtype = arr.dtype
        return result

    @classmethod
    def from_rimg(cls,rimg, dtype=float):
        img = cls(int(rimg.w),int(rimg.h),int(rimg.nchannels))
        img.dtype = dtype
        img.__set_data(np.ravel(np.array(rimg.get_data(),float)))
        return img
        
    def swap_back(self):
        self.data = list(np.ravel(self.rimg.get_data()).astype(float))  
        
    def to_array(self):
        
        self.swap_back()
        
        return np.array(self.data,float).reshape((self.h, self.w, self.nchannels)).astype(self.dtype if self.dtype is not None else float)
    
    def get_pixel(self, x, y, default_value=(0,0,0)):
        return np.array(list(self.rimg.get_pixel(x, y, rustcore.Pixel().push_vec([float(cv) for cv in default_value])).get_values())).astype(self.dtype if self.dtype is not None else float)

    def set_pixel(self, x, y, value):
        self.rimg.set_pixel(x,y,rustcore.Pixel().push_vec(value))

    def get_window(self, x, y, w, h, background=(0,0,0)):
        return Image.from_rimg(self.rimg.get_window(x,y,w,h,rustcore.Pixel().push_vec(background)),self.dtype if self.dtype is not None else float)

    def set_window(self, x, y, window):
        self.rimg.set_window(x,y,window.rimg)
        