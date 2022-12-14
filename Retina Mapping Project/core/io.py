import PIL.Image
from .path import normalize_path, init_folder
import utils as U
import numpy as np

def imsave(pixels, path):
    """! Output RGB pixels to a given path
    """

    name = normalize_path(path,make_absolute=True)

    U.dprint(f"Saving output image {name}...")

    if "/" in name:
        dirname = "/".join(name.split("/")[:-1])
        init_folder(dirname, clear=False)
        
    PIL.Image.fromarray(pixels).save(f"{name}.png")

    U.dprint("Done.")
    
def imload(path):
    return np.asarray(PIL.Image.open(path))

