# Import necessary libraries

import PIL.Image
import numpy as np


# Image IO Functions

def imload(path):
    return np.asarray(PIL.Image.open(path))


def imsave(pixels,path):
    PIL.Image.fromarray(pixels).save(path)