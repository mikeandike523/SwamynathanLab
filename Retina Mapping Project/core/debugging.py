from .path import init_folder, normalize_path

import PIL.Image

import utils as U

import os

import pprint

import termcolor
from termcolor import cprint

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

class Logging:

    COLOR_MAP = {
        "log": "white",
        "error": "red",
        "warn": "yellow",
        "info": "blue",
        "success": "green",
    }

    def __init__(self, tag=""):

        self.tag = tag

        self.prefix = f"[{self.tag}]: " if self.tag else ""

        # get the width of the terminal in columns
        columns = os.get_terminal_size().columns 

        self.pp = pprint.PrettyPrinter(indent=4, width=columns, compact=False)


    def print(self, level, obj):
        formatted = self.pp.pformat(obj).replace("\r\n","\n")
        formatted = "\n".join([self.prefix + line for line in formatted.split("\n")])
        cprint(self.prefix + formatted, Logging.COLOR_MAP[level])

    def __getattr__(self,attr_name):

        if attr_name in Logging.COLOR_MAP:
            return lambda message: self.print(attr_name, message)
        else:
            return object.__getattribute__(self, attr_name)
 
for k, v in Logging.COLOR_MAP.items():
    def logging_static_print(obj):
        logging = Logging()
        logging.print(k, obj)
    setattr(Logging, k, logging_static_print)
            



    

        


    
