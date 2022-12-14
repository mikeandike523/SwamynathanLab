from core import setup
setup.setup()

import os

import numpy as np
import PIL.Image

from movie import export_mp4_from_images

image_filenames = [fn for fn in os.listdir('output/mid_training_validation') if fn.endswith(".png")]

image_data = []

for image_filename in image_filenames:
    image_data.append(np.asarray(PIL.Image.open(f'output/mid_training_validation/{image_filename}')))

export_mp4_from_images(image_data,15,"output/mid_training_validation_movie.mp4")