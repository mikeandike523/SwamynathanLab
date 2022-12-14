from .path import init_folder, normalize_path

import PIL.Image

import utils as U

import os

def imshow(pixels, name="Untitled"):
    """! Output RGB pixels to debug/imshow/{name}.png
    """

    name = normalize_path(name)

    init_folder("debug/imshow",clear=False)

    U.dprint(f"Saving debug image {name}...")

    if "/" in name:
        dirname = "/".join(name.split("/")[:-1])
        init_folder(dirname, clear=False)
        
    PIL.Image.fromarray(pixels).save(f"debug/imshow/{name}.png")

    U.dprint("Done.")
    
class SequentialNames:
    
    def __init__(self, base_name):
        
        self.base_name = base_name
        
        self.counter = 0
        
    def next(self):

        retval = f"{self.base_name}.{self.counter}"

        self.counter +=  1
        
        return retval
            
    def current(self): # Used if one pipeline step needs to refer to the same name more than once   
            
        return f"{self.base_name}.{self.counter-1}"   
        
    def previous(self):    
            
        return f"{self.base_name}.{self.counter-2}"