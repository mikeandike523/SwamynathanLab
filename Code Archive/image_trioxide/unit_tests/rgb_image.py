import os
import sys

sys.path.insert(0,os.getcwd())

import io3

import numpy as np
import PIL.Image

from unit_testing import imsave, imload


pixels = imload("unit_tests/assets/download.jfif")

imsave(pixels, "unit_tests/output/original.png")

image = io3.RGBImage.from_array(pixels)

image = image.call_plugin("invert")

pixels = image.data

imsave(pixels, "unit_tests/output/inverted.png")
