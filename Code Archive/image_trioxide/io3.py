# Enable rust backtraces.
# @todo: consider adding feature to turn on/off backtraces for the image_trixoide library end-user
import os
os.environ["RUST_BACKTRACE"]="1"


# Import all pyO3 exported rust submodules here
# At this point there is only rgb_image but more may be added in the future
from image_trioxide import rgb_image

import numpy as np
import PIL.Image

import json


class RGBImage:
    """Unary operations will be instance methods."""
    """Binary and variadic operations will be class methods."""
    
    RGBImageClassPlugins = {
    }


    RGBImageInstancePlugins = {
        "invert":rgb_image.invert
    }


    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.pixels = np.zeros((h,w,3))
        
    @classmethod
    def from_array(cls, arr):
        rgb_image = cls(arr.shape[1],arr.shape[0])
        rgb_image.data = arr.copy()
        return rgb_image
    
    def to_json(self):

        root = {}
        root["w"] = self.w
        root["h"] = self.h
        root["data"] = [list(item) for item in self.data.astype(float).reshape((-1,3))]
        
        print("Encoding json...")
        
        retval = json.dumps(root)
        
        print("Done.")
        
        return retval
        
    @classmethod
    def from_json(cls, json_string):
        
        print(f"Decoding json...")
        
        root = json.loads(json_string)
        w = int(root["w"])
        h = int(root["h"])
        data = np.array(root["data"],np.uint8).reshape((h,w,3))
        
        print("Done.")
        
        return RGBImage.from_array(data)
    

    def call_plugin(self, plugin_name): # does not operate in-place
        plugin_input_string = self.to_json()
        print("Running plugin...")
        plugin_retval = RGBImage.RGBImageInstancePlugins[plugin_name](plugin_input_string)
        print("Done.")
        retval = RGBImage.from_json(plugin_retval)
        return retval
        